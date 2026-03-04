#!/usr/bin/env python3
# ROS1 (rospy) plane-based extrinsic calibration for a 2D spinning LiDAR on a robot TCP.
# No tf_transformations; uses SciPy Rotation + numpy only.

import sys, time, math
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from typing import List

import rospy
from sensor_msgs.msg import LaserScan
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rsc

# ---------- small helpers ----------
def rotvec_to_R(rvec):
    theta = norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def make_T(Rm, t):
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3, 3] = t.reshape(3)
    return T

def ransac_line_2d(points_xy, iters=300, thresh=0.004, min_inliers=80):
    if points_xy is None or len(points_xy) < 2:
        return None
    best = None
    n = points_xy.shape[0]
    rng = np.random.default_rng(123)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        p1, p2 = points_xy[i], points_xy[j]
        v = p2 - p1
        if norm(v) < 1e-6:
            continue
        nrm = np.array([v[1], -v[0]])
        nrmn = norm(nrm)
        if nrmn < 1e-9:
            continue
        a, b = nrm / nrmn
        c = -(a*p1[0] + b*p1[1])
        d = np.abs(a*points_xy[:,0] + b*points_xy[:,1] + c)
        inliers = d < thresh
        cnt = int(np.count_nonzero(inliers))
        if cnt >= min_inliers and (best is None or cnt > best[0]):
            best = (cnt, a, b, c, inliers)
    if best is None:
        return None
    _, a, b, c, inliers = best
    return a, b, c, inliers

@dataclass
class PoseCapture:
    A_bt: np.ndarray          # 4x4 base->TCP
    pts_lidar: np.ndarray     # Nx3 in LiDAR frame (z=0)

class CalibNode:
    def __init__(self):
        rospy.init_node("lidar_tcp_calibrator_ros1")

        self.lidar_topic  = rospy.get_param("~lidar_topic", "/scan")
        self.base_frame   = rospy.get_param("~base_frame", "base")
        self.tcp_frame    = rospy.get_param("~tcp_frame", "tool0")  # or ee_link
        self.angle_min_d  = float(rospy.get_param("~angle_min_deg", 135.0))
        self.angle_max_d  = float(rospy.get_param("~angle_max_deg", 225.0))
        self.scans_per_pose = int(rospy.get_param("~scans_per_pose", 3))
        self.min_range    = float(rospy.get_param("~min_range", 0.05))
        self.max_range    = float(rospy.get_param("~max_range", 8.0))
        self.poses_to_collect = int(rospy.get_param("~poses_to_collect", 20))
        self.ransac_thresh = float(rospy.get_param("~ransac_thresh", 0.004))
        self.min_inliers  = int(rospy.get_param("~min_inliers", 80))

        self.last_scan = None
        self.captures: List[PoseCapture] = []

        self.scan_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self.on_scan, queue_size=3)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("Ready. Move robot to a pose, then press ENTER in this terminal to capture.")
        rospy.loginfo(f"Config: topic={self.lidar_topic}, frames {self.base_frame}->{self.tcp_frame}, "
                      f"sector=[{self.angle_min_d},{self.angle_max_d}] deg, scans_per_pose={self.scans_per_pose}")

    def on_scan(self, msg: LaserScan):
        self.last_scan = msg

    def lookup_T_base_tcp(self, timeout=2.0):
        t_end = rospy.Time.now() + rospy.Duration(timeout)
        ex = None
        while rospy.Time.now() < t_end and not rospy.is_shutdown():
            try:
                tr: TransformStamped = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tcp_frame, rospy.Time(0), rospy.Duration(0.2)
                )
                t = tr.transform.translation
                q = tr.transform.rotation  # xyzw
                # Use SciPy Rotation to get matrix (no tf_transformations)
                Rm = Rsc.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T = np.eye(4)
                T[:3,:3] = Rm
                T[:3, 3] = np.array([t.x, t.y, t.z])
                return T
            except Exception as e:
                ex = e
                rospy.sleep(0.02)
        raise RuntimeError(f"TF {self.base_frame}->{self.tcp_frame} unavailable: {ex}")

    def extract_sector_points(self, scan: LaserScan):
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        ang_deg = (np.rad2deg(ang) + 360.0) % 360.0
        amin, amax = self.angle_min_d % 360.0, self.angle_max_d % 360.0
        if amin <= amax:
            mask = (ang_deg >= amin) & (ang_deg <= amax)
        else:
            mask = (ang_deg >= amin) | (ang_deg <= amax)
        r = np.array(scan.ranges, dtype=float)
        val = np.isfinite(r) & (r > self.min_range) & (r < self.max_range)
        keep = mask & val
        if not np.any(keep):
            return None
        th = ang[keep]; rr = r[keep]
        x = rr * np.cos(th); y = rr * np.sin(th)
        return np.stack([x, y], axis=1)

    def capture_pose(self):
        A_bt = self.lookup_T_base_tcp()

        collected = []
        need = self.scans_per_pose
        t0 = time.time()
        raw_count = 0
        scan_idx = 0

        while need > 0 and (time.time() - t0) < 5.0 and not rospy.is_shutdown():
            if self.last_scan is None:
                rospy.sleep(0.05)
                continue

            xy = self.extract_sector_points(self.last_scan)
            scan_idx += 1
            if xy is not None:
                raw_count += xy.shape[0]
                print(f"[scan {scan_idx}] sector points: {xy.shape[0]}", flush=True)

                # Only keep scans with enough points to be useful
                if xy.shape[0] > 10:
                    collected.append(xy)
                    need -= 1
            else:
                print(f"[scan {scan_idx}] sector points: 0", flush=True)

            rospy.sleep(0.05)

        if not collected:
            print("No scan captured; adjust pose/sector and try again.", flush=True)
            return False

        xy = np.vstack(collected)

        fit = ransac_line_2d(
            xy, iters=300, thresh=self.ransac_thresh, min_inliers=self.min_inliers
        )
        if fit is None:
            print("RANSAC failed; move/retarget and try again.", flush=True)
            return False

        a, b, c, inl = fit
        inl_xy = xy[inl]
        pts = np.hstack([inl_xy, np.zeros((inl_xy.shape[0], 1))])  # z=0 in lidar frame
        self.captures.append(PoseCapture(A_bt=A_bt, pts_lidar=pts))

        pose_idx = len(self.captures)
        print(
            f"Pose {pose_idx}: total raw sector points (all kept scans) = {raw_count} → "
            f"inliers kept = {pts.shape[0]}",
            flush=True
        )
        return True

    def solve(self):
        if len(self.captures) < 6:
            rospy.logerr("Need at least 6 captures.")
            return
        A_list = [c.A_bt for c in self.captures]
        P_list = [c.pts_lidar for c in self.captures]

        # x = [rvec(3), t(3), n(3), d(1)]
        x0 = np.zeros(10)
        x0[3:6] = np.array([0.0, 0.0, 0.10])  # 10 cm guess along TCP z
        x0[6:9] = np.array([0.0, 0.0, 1.0])

        def residuals(x):
            rvec = x[0:3]; tvec = x[3:6]; nvec = x[6:9]; d = x[9]
            n = nvec / (norm(nvec) + 1e-12)
            Rm = rotvec_to_R(rvec)
            T_tl = make_T(Rm, tvec)
            res = []
            for A_bt, pts in zip(A_list, P_list):
                pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
                pts_b = (A_bt @ (T_tl @ pts_h.T)).T[:, :3]
                res.append(pts_b @ n + d)
            return np.concatenate(res)

        sol = least_squares(residuals, x0, loss='cauchy', f_scale=0.01, max_nfev=200)
        x = sol.x
        rvec, tvec, nvec, d = x[0:3], x[3:6], x[6:9], x[9]
        n = nvec / (norm(nvec) + 1e-12)
        Rm = rotvec_to_R(rvec)
        T_tl = make_T(Rm, tvec)
        rms = math.sqrt(np.mean(sol.fun**2))
        rpy = Rsc.from_matrix(Rm).as_euler('xyz', degrees=False)

        print("\n================ Calibration Results (ROS1) ================")
        print("T_TCP_LiDAR (4x4):\n", np.array2string(T_tl, formatter={'float_kind':lambda v: f"{v: .6f}"}))
        print("Translation (m):", tvec)
        print("Rotation (rotvec, rad):", rvec)
        print(f"Rotation (RPY rad): roll={rpy[0]:.6f}, pitch={rpy[1]:.6f}, yaw={rpy[2]:.6f}")
        print("Plane normal n_b:", n)
        print(f"Plane offset d_b: {d:.6f}")
        print(f"Residual RMS (m): {rms:.6f}")
        print(f"Solver success: {sol.success}, iters: {sol.nfev}, status: {sol.status}")
        print("============================================================\n")

def main():
    node = CalibNode()
    try:
        while not rospy.is_shutdown():
            remaining = node.poses_to_collect - len(node.captures)
            if remaining <= 0:
                break
            sys.stdout.write(f"\nPress ENTER to capture (remaining {remaining}) or type 'solve' > ")
            sys.stdout.flush()
            s = sys.stdin.readline().strip()
            if s.lower() == 'solve':
                break
            ok = node.capture_pose()
            if not ok:
                print("Capture failed; adjust and try again.")
        node.solve()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()