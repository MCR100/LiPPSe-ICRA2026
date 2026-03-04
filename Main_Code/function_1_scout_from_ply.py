#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function-1 (PLY version): load existing torso point cloud (BASE), then:
- cluster cleanup on XZ,
- extremal Z -> lines -> N,
- rays @22.5° and 67.5° from N,
- +5 cm outward offsets,
- orientations / quaternions aligned to rays,
- save poses JSON and PNG,
- return same dict as before.

Usage:
    out = run_scout_from_ply(
            ply_path="updated_point_cloud.ply",
            save_json=True,
            json_path="/tmp/scout_two_start_poses.json",
            save_fig=True)

Notes:
- Expects ASCII PLY with x y z columns (exactly what Function 5 writes).
- All math is unchanged from your original Function 1; only the scan step is removed.
"""

import os, json, datetime
import numpy as np
import rospy

# headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- USER TUNABLES -----------------
GRID_M          = 0.008         # 8 mm cells for largest-component filter (XZ)
MIN_CELLS_TORSO = 150
RAY_OFFSET_M    = 0.07          # +5 cm outward along each ray
DEFAULT_JSON_PATH = "/tmp/scout_two_start_poses.json"
DEFAULT_PLY_PATH  = "updated_point_cloud.ply"   # <— read this file (in BASE)

# ----------------- MATH HELPERS -----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v / (n if n > 1e-12 else 1.0)

def frame_from_x_and_up(x_axis, up_axis):
    """Build right-handed frame with x along x_axis and 'up' as approximate z; returns 3x3 (columns=x,y,z)."""
    x = normalize(x_axis)
    y = np.cross(up_axis, x)
    if np.linalg.norm(y) < 1e-9:
        y = np.cross(np.array([0.0, 1.0, 0.0]), x)
    y = normalize(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, normalize(z)])

def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
            w = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
            w = (R[0,2] - R[2,0]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
            w = (R[1,0] - R[0,1]) / s
    q = np.array([x, y, z, w], dtype=float)
    n = np.linalg.norm(q) or 1.0
    return (q / n)

# ----------------- SIMPLE ASCII PLY READER (xyz only) -----------------
def read_ply_xyz(path):
    """
    Minimal ASCII PLY (xyz) reader matching Function 5's writer.
    Returns Nx3 float32 array.
    """
    with open(path, "r") as f:
        header = []
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a PLY file")
        header.append(line)
        nverts = None
        has_color = False
        # parse header
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex"):
                nverts = int(line.split()[-1])
            if line.startswith("property uchar"):
                has_color = True
            if line == "end_header":
                break
        if nverts is None:
            raise ValueError("PLY missing 'element vertex'")
        # read body
        pts = np.zeros((nverts, 3), dtype=np.float32)
        for i in range(nverts):
            row = f.readline()
            if row is None:
                raise ValueError("PLY truncated")
            parts = row.strip().split()
            if len(parts) < 3:
                raise ValueError("PLY vertex line has <3 columns")
            pts[i, 0] = float(parts[0])
            pts[i, 1] = float(parts[1])
            pts[i, 2] = float(parts[2])
        return pts

# ----------------- 2D GRID CLUSTERING (on XZ) -----------------
def largest_component_mask_xz(Pxz: np.ndarray, grid=GRID_M, min_cells=MIN_CELLS_TORSO):
    if Pxz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x, z = Pxz[:,0], Pxz[:,1]
    x0, z0 = x.min(), z.min()
    xi = np.floor((x - x0) / grid).astype(np.int32)
    zi = np.floor((z - z0) / grid).astype(np.int32)
    cells, inv = np.unique(np.stack([xi, zi], axis=1), axis=0, return_inverse=True)
    n_cells = cells.shape[0]
    parent = np.arange(n_cells, dtype=np.int32)
    size   = np.ones(n_cells, dtype=np.int32)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if size[ra] < size[rb]: ra, rb = rb, ra
        parent[rb] = ra
        size[ra]  += size[rb]
    cell_to_idx = {(int(cx), int(cz)): idx for idx, (cx, cz) in enumerate(cells)}
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for idx, (cx, cz) in enumerate(cells):
        for dx, dz in neigh:
            j = cell_to_idx.get((int(cx+dx), int(cz+dz)))
            if j is not None:
                union(idx, j)
    roots = np.array([find(i) for i in range(n_cells)], dtype=np.int32)
    unique_roots, counts = np.unique(roots, return_counts=True)
    biggest_root = unique_roots[np.argmax(counts)]
    biggest_mask_cells = (roots == biggest_root)
    if counts.max() < min_cells:
        rospy.logwarn_throttle(2.0, f"[Scout] Largest component has only {counts.max()} cells (<{min_cells}). "
                                    f"Grid={grid:.3f} m may be too fine or scan too sparse.")
    mask_points = biggest_mask_cells[inv]
    return mask_points

# --------------- Ray helper ---------------
def farthest_on_ray(Pxz: np.ndarray, Nxz: np.ndarray, d: np.ndarray, base_tol_deg: float = 5.0):
    """Return (point_xz, idx) of farthest point along ray N + t d, t>0 (with widening angular tolerance)."""
    v = Pxz - Nxz[None, :]
    t = v @ d
    norm_v = np.linalg.norm(v, axis=1)
    cross = np.abs(v[:,0]*d[1] - v[:,1]*d[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        sinang = np.where(norm_v > 1e-9, cross / norm_v, np.inf)
    sinang = np.clip(sinang, 0.0, 1.0)
    ang = np.arcsin(sinang)
    for tol_deg in (base_tol_deg, 10.0, 15.0, 20.0, 30.0):
        tol = np.deg2rad(tol_deg)
        mask = (t > 0.0) & (ang <= tol)
        if np.any(mask):
            idxs = np.flatnonzero(mask)
            best = int(idxs[np.argmax(t[idxs])])
            return Pxz[best], best
    pos = np.where(t > 0.0)[0]
    if pos.size == 0:
        return Nxz.copy(), -1
    best = int(pos[np.argmax(t[pos])])
    return Pxz[best], best

# ----------------- CORE PROCESSOR -----------------
def process_point_cloud(P, grid_m=GRID_M, min_cells=MIN_CELLS_TORSO, ray_offset_m=RAY_OFFSET_M):
    """
    Takes Nx3 points in BASE, returns the same tuple as your original mapper.run().
    """
    rospy.loginfo(f"[Scout] Loaded {P.shape[0]} points before filtering.")

    # Cluster cleanup on (X,Z) to isolate torso
    Pxz = P[:, [0, 2]]
    mask = largest_component_mask_xz(Pxz, grid=grid_m, min_cells=min_cells)
    if mask.sum() == 0:
        rospy.logwarn("[Scout] Cluster filter kept 0 points; falling back to raw cloud.")
        P_clean = P
        Pxz_clean = Pxz
    else:
        P_clean = P[mask]
        Pxz_clean = Pxz[mask]

    rospy.loginfo(f"[Scout] Kept {P_clean.shape[0]} points in largest component.")

    # Extremal Z points in cleaned cloud
    z_vals = P_clean[:, 2]
    idx_z_max = int(np.argmax(z_vals))
    idx_z_min = int(np.argmin(z_vals))
    p_z_max = P_clean[idx_z_max]   # [x,y,z] (topmost)
    p_z_min = P_clean[idx_z_min]   # [x,y,z] (closest to bed)

    # Define N from x @ z_max and z @ z_min (vertical/horizontal lines intersection)
    x_maxZ = p_z_max[0]
    z_minZ = p_z_min[2]
    N = np.array([x_maxZ, 0.0, z_minZ], dtype=np.float64)

    # Rays from N in XZ plane at 22.5° and 67.5°, oriented toward torso (based on p_z_min vs N)
    Nxz = np.array([N[0], N[2]], dtype=np.float64)
    sign_x = np.sign(p_z_min[0] - N[0]) or 1.0
    ex = np.array([sign_x, 0.0])  # 0°
    ez = np.array([0.0, 1.0])     # 90°
    def dir_from_theta(theta_rad):
        d = np.cos(theta_rad)*ex + np.sin(theta_rad)*ez
        n = np.linalg.norm(d)
        return d / (n if n > 1e-9 else 1.0)

    d1 = dir_from_theta(np.deg2rad(22.5))
    d2 = dir_from_theta(np.deg2rad(67.5))

    # Intersections: farthest usable sample along each ray
    p1_xz, i1 = farthest_on_ray(Pxz_clean, Nxz, d1, base_tol_deg=5.0)
    p2_xz, i2 = farthest_on_ray(Pxz_clean, Nxz, d2, base_tol_deg=5.0)

    p1 = P_clean[i1] if i1 != -1 else np.array([p1_xz[0], 0.0, p1_xz[1]])
    p2 = P_clean[i2] if i2 != -1 else np.array([p2_xz[0], 0.0, p2_xz[1]])

    # +5 cm outward along the rays (keep Y from torso point)
    p1_out_xz = p1_xz + ray_offset_m * d1
    p2_out_xz = p2_xz + ray_offset_m * d2
    p1_out = np.array([p1_out_xz[0], p1[1], p1_out_xz[1]], dtype=np.float64)
    p2_out = np.array([p2_out_xz[0], p2[1], p2_out_xz[1]], dtype=np.float64)

    # Yaw angles of the rays in BASE XZ
    yaw1_deg = float(np.degrees(np.arctan2(d1[1], d1[0])))
    yaw2_deg = float(np.degrees(np.arctan2(d2[1], d2[0])))

    # Full BASE orientations at p1_out/p2_out
    Z_up = np.array([0.0, 0.0, 1.0])
    dir1_base = normalize(np.array([d1[0], 0.0, d1[1]]))
    dir2_base = normalize(np.array([d2[0], 0.0, d2[1]]))
    R1 = frame_from_x_and_up(dir1_base, Z_up)  # columns are tool x,y,z in BASE
    R2 = frame_from_x_and_up(dir2_base, Z_up)
    q1 = mat_to_quat(R1)
    q2 = mat_to_quat(R2)

    poses = {
        "p1_out": {"position_base_m": p1_out.tolist(),
                   "R_base_from_tcp": R1.tolist(),
                   "quaternion_xyzw": q1.tolist(),
                   "ray_yaw_xz_deg": yaw1_deg},
        "p2_out": {"position_base_m": p2_out.tolist(),
                   "R_base_from_tcp": R2.tolist(),
                   "quaternion_xyzw": q2.tolist(),
                   "ray_yaw_xz_deg": yaw2_deg},
        "N_frame": {"N_base": [float(N[0]), float(N[1]), float(N[2])],
                    "angles_deg": [22.5, 67.5]}
    }

    return P, P_clean, p_z_max, p_z_min, N, p1, p2, p1_out, p2_out, yaw1_deg, yaw2_deg, R1, R2, q1, q2, poses

# ----------------- PUBLIC CALLABLE -----------------
def run_scout_from_ply(ply_path=DEFAULT_PLY_PATH,
                       save_json=True,
                       json_path=DEFAULT_JSON_PATH,
                       save_fig=True,
                       fig_dir=None,
                       fig_prefix="scout_slice_from_ply"):
    """
    Load PLY (BASE), compute two start poses, save JSON/PNG, and return results.
    Returns dict:
      p1, p2, p1_out, p2_out, yaw1_deg, yaw2_deg, q1_xyzw, q2_xyzw, R1, R2,
      poses_json, json_path, fig_path
    """
    # Ensure ROS node
    try:
        import rospy.core as roscore
        if not roscore.is_initialized():
            rospy.init_node("scout_from_ply", anonymous=True)
    except Exception:
        pass

    if not os.path.isfile(ply_path):
        rospy.logerr(f"[Scout] PLY not found: {ply_path}")
        return None

    # Load PLY
    P = read_ply_xyz(ply_path)

    # Process
    (P_all, P_clean, p_z_max, p_z_min, N, p1, p2, p1_out, p2_out,
     yaw1_deg, yaw2_deg, R1, R2, q1, q2, poses) = process_point_cloud(P)

    # Save JSON
    saved_json_path = None
    if save_json and poses is not None:
        try:
            os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(poses, f, indent=2)
            rospy.loginfo(f"[Scout] Saved poses JSON -> {json_path}")
            saved_json_path = json_path
        except Exception as e:
            rospy.logwarn(f"[Scout] Failed to save JSON: {e}")

    # Save PNG (plot CLEAN cloud, not raw)
    saved_fig_path = None
    if save_fig and (P_clean is not None) and (P_clean.shape[0] > 0):
        try:
            fig_dir = fig_dir or os.getcwd()
            os.makedirs(fig_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(fig_dir, f"{fig_prefix}_{ts}.png")

            x = P_clean[:,0]; z = P_clean[:,2]
            plt.figure(figsize=(7,6))
            plt.scatter(x, z, s=1, label="torso (clean)")

            x_min, x_max = float(np.min(x)), float(np.max(x))
            z_min, z_max = float(np.min(z)), float(np.max(z))
            plt.hlines(y=p_z_min[2], xmin=x_min, xmax=x_max, linestyles="--", alpha=0.5, label="z=min line")
            plt.vlines(x=p_z_max[0], ymin=z_min, ymax=z_max, linestyles="--", alpha=0.5, label="x at z=max")

            plt.scatter([p_z_max[0]],[p_z_max[2]], s=40, marker='o', label="max Z")
            plt.scatter([p_z_min[0]],[p_z_min[2]], s=40, marker='x', label="min Z")
            plt.scatter([N[0]],[N[2]], s=45, marker='^', label="N")

            plt.plot([N[0], p1[0]], [N[2], p1[2]], linewidth=1.5, label="ray 22.5°")
            plt.plot([N[0], p2[0]], [N[2], p2[2]], linewidth=1.5, label="ray 67.5°")
            plt.plot([p1[0], p1_out[0]], [p1[2], p1_out[2]], linewidth=2.0)
            plt.plot([p2[0], p2_out[0]], [p2[2], p2_out[2]], linewidth=2.0)

            plt.scatter([p1[0]],[p1[2]], s=45, marker='s', label="p1")
            plt.scatter([p2[0]],[p2[2]], s=45, marker='D', label="p2")
            plt.scatter([p1_out[0]],[p1_out[2]], s=50, marker='P', label="p1_out")
            plt.scatter([p2_out[0]],[p2_out[2]], s=50, marker='X', label="p2_out")

            plt.xlabel("base X (m)")
            plt.ylabel("base Z (m)")
            plt.title("Scout from PLY (BASE): N rays, +5 cm offsets, full poses")
            plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
            rospy.loginfo(f"[Scout] Saved figure PNG -> {fig_path}")
            saved_fig_path = fig_path
        except Exception as e:
            rospy.logwarn(f"[Scout] Failed to save PNG: {e}")

    return {
        "p1": p1, "p2": p2,
        "p1_out": p1_out, "p2_out": p2_out,
        "yaw1_deg": yaw1_deg, "yaw2_deg": yaw2_deg,
        "q1_xyzw": q1, "q2_xyzw": q2,
        "R1": R1, "R2": R2,
        "poses_json": poses,
        "json_path": saved_json_path,
        "fig_path": saved_fig_path,
    }

# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    try:
        out = run_scout_from_ply(
            ply_path=DEFAULT_PLY_PATH,
            save_json=True,
            json_path=DEFAULT_JSON_PATH,
            save_fig=True,
            fig_dir=None
        )
        if out is not None:
            print("\n=== p1_out (+5cm along 22.5°) ===")
            print("pos [m]:", out["p1_out"])
            print("quat (x,y,z,w):", out["q1_xyzw"])
            print(f"yaw_xz [deg wrt +X]: {out['yaw1_deg']:.2f}")

            print("\n=== p2_out (+5cm along 67.5°) ===")
            print("pos [m]:", out["p2_out"])
            print("quat (x,y,z,w):", out["q2_xyzw"])
            print(f"yaw_xz [deg wrt +X]: {out['yaw2_deg']:.2f}")

            print("\nSaved JSON:", out["json_path"])
            print("Saved PNG :", out["fig_path"])
    except rospy.ROSInterruptException:
        pass