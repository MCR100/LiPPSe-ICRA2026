#!/usr/bin/env python3
"""
Map RPLIDAR 2D scans to robot base frame and save a 3D PLY.

ROS1 (rospy) version. No external deps beyond numpy.
- Subscribes: sensor_msgs/LaserScan on /scan
- TF: base -> tcp_frame
- Uses calibrated T_TCP->LiDAR (fill in below)
- Restricts beams to [angle_min_deg, angle_max_deg] (default 135..225)
- Duration-controlled capture (default 40 s) OR stop via service ~/stop
- Writes ASCII PLY at the end
"""

import math
import sys
import time
from typing import List, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerResponse
import datetime

# ----------------- USER PARAMETERS (edit here or via ROS params) -----------------

# Frames and topics
BASE_FRAME = "base"
TCP_FRAME = "tool0"            # or "ee_link"
LIDAR_TOPIC = "/scan"

# Capture settings
DURATION_SEC = 10.0            # total capture time (used as a safety cap)
ANGLE_MIN_DEG = 135.0          # sector start (deg)
ANGLE_MAX_DEG = 225.0          # sector end (deg)
MIN_RANGE = 0.05               # m
MAX_RANGE = 0.50               # m
DESKEW_MODE = "per_beam"       # "per_scan" or "per_beam"
MAX_POINTS = 2_000_000         # safety cap

# Calibrated T_TCP->LiDAR (4x4) from your solve (example values; replace with yours)
T_TCP_LIDAR = np.array([
    [-0.030898, -0.002688,  0.999519,  0.066865],
    [-0.030407,  0.999536,  0.001748, -0.001483],
    [-0.999060, -0.030338, -0.030966,  0.033065],
    [ 0.0     ,  0.0     ,  0.0     ,  1.0     ]
], dtype=np.float64)

# Output
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#OUTPUT_PLY_PATH = f"scan3d_output_{timestamp_str}.ply"
OUTPUT_PLY_PATH = f"updated_point_cloud.ply"

# --------------------------- helpers ---------------------------

def homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """Nx3 -> Nx4 homogeneous"""
    return np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)])

def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    pts_h = homogeneous(pts_xyz)            # Nx4
    out = (T @ pts_h.T).T                   # Nx4
    return out[:, :3]

def angle_mask(angles_rad: np.ndarray, a0_deg: float, a1_deg: float) -> np.ndarray:
    """Boolean mask for angles within [a0_deg, a1_deg] modulo 360."""
    ang_deg = (np.rad2deg(angles_rad) + 360.0) % 360.0
    a0 = a0_deg % 360.0
    a1 = a1_deg % 360.0
    if a0 <= a1:
        return (ang_deg >= a0) & (ang_deg <= a1)
    else:
        return (ang_deg >= a0) | (ang_deg <= a1)

def lookup_T(tf_buffer: tf2_ros.Buffer, parent: str, child: str, t: rospy.Time, timeout=0.2) -> np.ndarray:
    """TF lookup as 4x4 matrix (parent->child)."""
    tr: TransformStamped = tf_buffer.lookup_transform(parent, child, t, rospy.Duration(timeout))
    tx = tr.transform.translation
    q = tr.transform.rotation
    # rotation matrix from quaternion (x,y,z,w)
    x, y, z, w = q.x, q.y, q.z, q.w
    R = quat_to_mat(x, y, z, w)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx.x, tx.y, tx.z], dtype=np.float64)
    return T

def quat_to_mat(x, y, z, w) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1.0 - 2.0*(yy+zz),     2.0*(xy - wz),         2.0*(xz + wy)],
        [    2.0*(xy + wz), 1.0 - 2.0*(xx+zz),         2.0*(yz - wx)],
        [    2.0*(xz - wy),     2.0*(yz + wx),     1.0 - 2.0*(xx+yy)]
    ], dtype=np.float64)

def write_ply_ascii(path: str, pts: np.ndarray, colors: np.ndarray = None):
    """Write ASCII PLY. pts Nx3, optional colors Nx3 uint8."""
    n = pts.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# --------------------------- main capture ---------------------------

class Mapper:
    def __init__(self):
        rospy.init_node("lidar_to_base_mapper", anonymous=False)

        # Allow overriding via ROS params if you want
        self.base_frame   = rospy.get_param("~base_frame", BASE_FRAME)
        self.tcp_frame    = rospy.get_param("~tcp_frame",  TCP_FRAME)
        self.topic        = rospy.get_param("~lidar_topic", LIDAR_TOPIC)
        self.duration_sec = float(rospy.get_param("~duration_sec", DURATION_SEC))
        self.amin_deg     = float(rospy.get_param("~angle_min_deg", ANGLE_MIN_DEG))
        self.amax_deg     = float(rospy.get_param("~angle_max_deg", ANGLE_MAX_DEG))
        self.min_range    = float(rospy.get_param("~min_range", MIN_RANGE))
        self.max_range    = float(rospy.get_param("~max_range", MAX_RANGE))
        self.deskew_mode  = str(rospy.get_param("~deskew_mode", DESKEW_MODE))
        self.max_points   = int(rospy.get_param("~max_points", MAX_POINTS))
        self.output_path  = str(rospy.get_param("~output_ply", OUTPUT_PLY_PATH))

        # Static calibrated T_TCP->LiDAR
        self.T_tcp_lidar  = T_TCP_LIDAR.copy()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.points_base: List[np.ndarray] = []
        self.total_points = 0

        # --- Stop on command support ---
        self.stop_requested = False
        self.stop_srv = rospy.Service("~stop", Trigger, self._handle_stop)

        rospy.Subscriber(self.topic, LaserScan, self.scan_cb, queue_size=5)

        rospy.loginfo(f"Starting capture for up to {self.duration_sec:.1f}s "
                      f"(sector {self.amin_deg}..{self.amax_deg} deg, mode {self.deskew_mode}).")
        self.t_start = rospy.Time.now()

    def _handle_stop(self, req):
        """Service handler to request an early stop."""
        self.stop_requested = True
        return TriggerResponse(success=True, message="Stop requested")

    def scan_cb(self, scan: LaserScan):
        # Stop immediately if requested
        if self.stop_requested:
            return

        # Duration guard (still acts as a safety cap)
        if (rospy.Time.now() - self.t_start).to_sec() > self.duration_sec:
            return

        # Angles for this scan
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment
        mask_sector = angle_mask(angles, self.amin_deg, self.amax_deg)
        if not np.any(mask_sector):
            return

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > self.min_range) & (ranges < self.max_range)
        keep = mask_sector & valid
        if not np.any(keep):
            return

        th = angles[keep]
        rr = ranges[keep]
        # 2D points in LiDAR frame (z=0)
        x_l = rr * np.cos(th)
        y_l = rr * np.sin(th)
        z_l = np.zeros_like(x_l)
        pts_l = np.stack([x_l, y_l, z_l], axis=1)  # Nx3

        # Deskew
        try:
            if self.deskew_mode == "per_beam" and scan.time_increment > 0.0:
                # Per-beam TF (precise)
                pts_b = self._transform_per_beam(scan, keep, pts_l)
            else:
                # Per-scan TF (fast)
                T_bt = lookup_T(self.tf_buffer, self.base_frame, self.tcp_frame, scan.header.stamp)
                T = T_bt @ self.T_tcp_lidar
                pts_b = transform_points(T, pts_l)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"TF lookup failed: {e}")
            return

        # Append (cap to avoid runaway memory)
        if self.total_points + pts_b.shape[0] > self.max_points:
            remain = self.max_points - self.total_points
            if remain <= 0:
                return
            pts_b = pts_b[:remain, :]
        self.points_base.append(pts_b)
        self.total_points += pts_b.shape[0]

        # Optional: print some progress
        if self.total_points % 50000 < pts_b.shape[0]:
            rospy.loginfo(f"Accumulated {self.total_points} points...")

    def _transform_per_beam(self, scan: LaserScan, keep_mask: np.ndarray, pts_l: np.ndarray) -> np.ndarray:
        """
        Precise deskew: for each kept beam, compute its timestamp inside the scan,
        look up base->TCP at that time, then transform.
        """
        idxs = np.nonzero(keep_mask)[0]
        t0 = scan.header.stamp
        dt = scan.time_increment  # seconds per beam
        pts_out = np.empty_like(pts_l)

        # Pre-multiply with T_tcp_lidar for speed
        pts_l_h = homogeneous(pts_l).T  # 4xN
        T_tl = self.T_tcp_lidar

        for k, i in enumerate(idxs):
            if self.stop_requested:
                break
            beam_time = t0 + rospy.Duration.from_sec(i * dt)
            T_bt = lookup_T(self.tf_buffer, self.base_frame, self.tcp_frame, beam_time)
            T = T_bt @ T_tl
            pb = (T @ pts_l_h[:, k]).ravel()[:3]
            pts_out[k, :] = pb
        return pts_out

    def run(self):
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown():
            if self.stop_requested:
                rospy.loginfo("Stop requested; finishing up…")
                break
            elapsed = (rospy.Time.now() - self.t_start).to_sec()
            if elapsed >= self.duration_sec:
                rospy.loginfo("Duration reached; finishing up…")
                break
            rate.sleep()

        # Concatenate and write PLY
        if len(self.points_base) == 0:
            rospy.logwarn("No points captured; nothing to save.")
            return

        pts = np.vstack(self.points_base)
        rospy.loginfo(f"Saving {pts.shape[0]} points to {self.output_path} ...")
        write_ply_ascii(self.output_path, pts)
        rospy.loginfo("Done.")

# --------------------------- callable wrapper ---------------------------

def run_mapper():
    """Run the Mapper capture (callable from another script)."""
    try:
        mapper = Mapper()
        mapper.run()
    except rospy.ROSInterruptException:
        pass

# --------------------------- entrypoint ---------------------------

if __name__ == "__main__":
    run_mapper()