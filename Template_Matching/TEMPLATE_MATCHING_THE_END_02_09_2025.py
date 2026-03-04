#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template matching (point cloud → point cloud) + correspondence via white marker,
then PCA normal estimation at the corresponding scene point and visualization.

- Scene = point cloud (already cleaned/downsampled upstream)
- No voxel downsampling here
- Marker color to find on template: WHITE (1,1,1)
- Visualization colors:
    Scene = red, Template = green, Template marker sphere = blue, NN sphere = orange
- FINAL VIEW: ONLY cropped_scene + PCA normal line + PCA plane patch
"""

import copy
import numpy as np
import open3d as o3d
from Voxel_Callable_Final_02_09_2025 import clean_torso_point_cloud

# ------------------- INPUTS -------------------
#PLY_IN="roshan.ply"
PLY_IN = r"E:\PYCHARM\PYCHARM_LIDAR\Evaluation_1\DATA\mauricio\predefined\1.ply"
pcd_final = clean_torso_point_cloud(PLY_IN, visualize=True)   # your processed scene PCD
processed_pcd = pcd_final
TEMPLATE_PCD_PATH = "template_colored_1.ply"                     # template PCD with a WHITE marker point

# ------------------- PARAMS -------------------
VOXEL_SIZE      = 0.005    # used only for normal/FPFH search radii
DIST_THRESH     = 0.075
FGR_ITERATIONS  = 50
FGR_MAX_TUPLES  = 1000
ICP_RADIUS      = 0.04
TARGET_FITNESS  = 0.80
WHITE_RGB       = np.array([1.0, 1.0, 1.0])

# PCA normal / plane vis
PCA_K           = 30       # k-NN size for PCA normal
NORMAL_SCALE    = 0.03     # length of the drawn normal line (m)
PLANE_SIZE      = 0.08     # side length of the plane patch (m)
PLANE_COLOR     = (0.9, 0.9, 0.2)  # light yellow (no transparency in default viewer)
# ------------------------------------------------

# ---------- Utility functions ----------
def export_points_to_txt(template_point_xyz, scene_point_xyz, filename="points.txt"):
    with open(filename, "w") as f:
        f.write(f"Template white point: {np.asarray(template_point_xyz)}\n")
        f.write(f"Corresponding scene point: {np.asarray(scene_point_xyz)}\n")
    print(f"[Save] Wrote points to {filename}")

def compute_normals_and_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float):
    if pcd.is_empty():
        raise SystemExit("ERROR: empty point cloud given to compute_normals_and_fpfh")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return pcd, fpfh

def cprmse(res) -> float:
    n = len(res.correspondence_set)
    return (res.inlier_rmse / n) if n != 0 else 1e3

def pca_normal_at_index(pcd: o3d.geometry.PointCloud, idx: int, k: int = 30) -> np.ndarray:
    """
    PCA/SVD plane fit on k-NN of point idx. Returns a unit normal vector.
    Orients the normal to point outward from the cloud centroid for consistency.
    """
    if idx < 0 or idx >= len(pcd.points):
        raise ValueError("Index out of bounds for PCA normal")

    kd = o3d.geometry.KDTreeFlann(pcd)
    _, nn_idx, _ = kd.search_knn_vector_3d(pcd.points[idx], max(3, k))
    P = np.asarray(pcd.points)[nn_idx, :]
    mu = P.mean(axis=0)
    C = (P - mu).T @ (P - mu) / P.shape[0]
    w, V = np.linalg.eigh(C)         # ascending eigenvalues
    n = V[:, 0]                      # eigenvector for smallest eigenvalue
    n = n / (np.linalg.norm(n) + 1e-12)

    # Consistent orientation: point normal away from the global centroid
    centroid = np.asarray(pcd.points).mean(axis=0)
    p_idx = np.asarray(pcd.points)[idx]
    if np.dot(n, (p_idx - centroid)) < 0:
        n = -n
    return n

def make_normal_lineset(origin: np.ndarray, direction: np.ndarray, scale: float = 0.03,
                        color=(0.8, 0.0, 0.8)) -> o3d.geometry.LineSet:
    """Create a simple line (origin -> origin + scale*direction) as a LineSet."""
    p0 = origin
    p1 = origin + scale * direction
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([p0, p1])
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])   # one line, one color
    return ls

def make_plane_patch(origin: np.ndarray, normal: np.ndarray, size: float = 0.08,
                     color=(0.9, 0.9, 0.2)) -> o3d.geometry.TriangleMesh:
    """
    Create a square plane patch (two triangles) centered at `origin`,
    oriented by `normal`, with side length `size`.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    # Build orthonormal basis (u, v, n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u);      v /= (np.linalg.norm(v) + 1e-12)

    h = size * 0.5
    p00 = origin - h*u - h*v
    p01 = origin - h*u + h*v
    p10 = origin + h*u - h*v
    p11 = origin + h*u + h*v

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector([p00, p01, p10, p11])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 1, 3]])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh
# ---------------------------------------


if __name__ == "__main__":
    # 1) Load scene & template (POINT CLOUDS)
    print("[Load] Reading point clouds...")
    scene = processed_pcd
    if scene.is_empty():
        raise SystemExit("Scene point cloud is empty after preprocessing.")

    template = o3d.io.read_point_cloud(TEMPLATE_PCD_PATH)
    if template.is_empty():
        raise SystemExit(f"Template point cloud not found/empty: {TEMPLATE_PCD_PATH}")

    # 2) Scaled template variants (±10%)
    print("[Prep] Building scaled template variants...")
    center = template.get_center()
    tmpl_small = copy.deepcopy(template); tmpl_small.scale(0.9, center=center)
    tmpl_large = copy.deepcopy(template); tmpl_large.scale(1.1, center=center)

    # 3) Normals + FPFH (no voxel downsample)
    print("[Prep] Computing normals + FPFH...")
    scene_proc,  scene_fpfh  = compute_normals_and_fpfh(scene, VOXEL_SIZE)
    tmpl_proc,   tmpl_fpfh   = compute_normals_and_fpfh(template, VOXEL_SIZE)
    tl_proc,     tl_fpfh     = compute_normals_and_fpfh(tmpl_large, VOXEL_SIZE)
    ts_proc,     ts_fpfh     = compute_normals_and_fpfh(tmpl_small, VOXEL_SIZE)

    # 4) FGR (large & small)
    print("[FGR] Running Fast Global Registration (large & small)...")
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=DIST_THRESH,
        iteration_number=FGR_ITERATIONS,
        maximum_tuple_count=FGR_MAX_TUPLES
    )
    res_large = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        tl_proc, scene_proc, tl_fpfh, scene_fpfh, opt
    )
    print(res_large, "\nfitness_large =", res_large.fitness, "\n")

    res_small = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        ts_proc, scene_proc, ts_fpfh, scene_fpfh, opt
    )
    print(res_small, "\nfitness_small =", res_small.fitness, "\n")

    # 5) Initial ICP (identity seed)
    print("[ICP] Initial ICP (identity seed)...")
    icp_initial = o3d.pipelines.registration.registration_icp(
        tmpl_proc, scene_proc, ICP_RADIUS, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    print("ICP initial fitness =", icp_initial.fitness, "\n")

    # Decide path by CPRMSE
    cpr_small  = cprmse(res_small)
    cpr_large  = cprmse(res_large)
    cpr_icp0   = cprmse(icp_initial)

    # 6) Enlarge/shrink loop
    current_fitness = 0.0
    transform = np.eye(4)
    template_to_use = tmpl_proc

    while current_fitness < TARGET_FITNESS:
        if (cpr_small > cpr_large) and ((cpr_icp0 > cpr_large) or (cpr_icp0 == 0)):
            print("[Loop] Enlarge path...")
            icp = o3d.pipelines.registration.registration_icp(
                tl_proc, scene_proc, ICP_RADIUS, res_large.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            current_fitness = icp.fitness
            print("  fitness (large) =", current_fitness)
            transform = icp.transformation
            template_to_use = tl_proc
            c = tl_proc.get_center()
            tl_proc.scale(1.1, center=c)
        elif (cpr_small < cpr_large) and ((cpr_icp0 > cpr_small) or (cpr_icp0 == 0)):
            print("[Loop] Shrink path...")
            icp = o3d.pipelines.registration.registration_icp(
                ts_proc, scene_proc, ICP_RADIUS, res_small.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            current_fitness = icp.fitness
            print("  fitness (small) =", current_fitness)
            transform = icp.transformation
            template_to_use = ts_proc
            c = ts_proc.get_center()
            ts_proc.scale(0.9, center=c)
        else:
            print("[Loop] Using initial ICP result...")
            current_fitness = icp_initial.fitness
            transform = icp_initial.transformation
            template_to_use = tmpl_proc

        if current_fitness >= TARGET_FITNESS:
            print(f"[Stop] Reached fitness {current_fitness:.3f} >= {TARGET_FITNESS}")
            break

    # 7) Apply final transform to template
    aligned_template = copy.deepcopy(template_to_use).transform(transform)

    # 8) Find WHITE marker point on aligned template
    if not aligned_template.has_colors():
        raise SystemExit("Aligned template has no colors; cannot find white marker.")
    colors_aligned = np.asarray(aligned_template.colors)
    white_idx = np.where(np.all(colors_aligned == WHITE_RGB, axis=1))[0]
    if white_idx.size == 0:
        raise SystemExit("No white-colored point found on template.")
    white_idx = int(white_idx[0])
    template_point = np.asarray(aligned_template.points)[white_idx]
    print("[Marker] Template white point index:", white_idx)

    # Visualization 1: global overlay (template = green, scene = red)
    scene_show = copy.deepcopy(scene_proc); scene_show.paint_uniform_color([1, 0, 0])      # red
    aligned_show = copy.deepcopy(aligned_template); aligned_show.paint_uniform_color([0, 1, 0])  # green
    o3d.visualization.draw_geometries([aligned_show, scene_show],
                                      window_name="Aligned template (green) vs scene (red)")

    # 9) Crop scene by aligned template AABB
    bbox = aligned_template.get_axis_aligned_bounding_box()
    bbox.color = np.array([0, 0, 1])  # blue AABB
    cropped_scene = scene_proc.crop(bbox)
    Pcrop = np.asarray(cropped_scene.points)
    if len(Pcrop) == 0:
        raise SystemExit("Cropped scene is empty; adjust alignment or crop bounds.")

    # Visualization 2: cropped scene + template (consistent colors)
    scene_cropped_show = copy.deepcopy(cropped_scene); scene_cropped_show.paint_uniform_color([1, 0, 0])  # red
    aligned_show2 = copy.deepcopy(aligned_template); aligned_show2.paint_uniform_color([0, 1, 0])         # green
    o3d.visualization.draw_geometries([aligned_show2, scene_cropped_show, bbox],
                                      window_name="Cropped scene (red) + template (green)")

    # 10) Nearest scene point (in cropped_scene) to template's white marker
    dists = np.linalg.norm(Pcrop - template_point, axis=1)
    nn_idx = int(np.argmin(dists))
    closest_point = Pcrop[nn_idx]
    print(f"[NN] Closest scene point index in cropped_scene = {nn_idx}")

    # ---- PCA normal at this corresponding scene point ----
    normal_vec = pca_normal_at_index(cropped_scene, nn_idx, k=PCA_K)
    print("[Normal/PCA] Unit normal at NN point:", normal_vec)

    # Build normal line and plane patch
    normal_line = make_normal_lineset(closest_point, normal_vec, scale=NORMAL_SCALE, color=(0.8, 0.0, 0.8))  # magenta
    plane_patch = make_plane_patch(closest_point, normal_vec, size=PLANE_SIZE, color=PLANE_COLOR)

    # Visualization 3: markers (white = template marker, orange = nearest scene point)
    template_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    template_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # blue
    template_sphere.translate(template_point)

    closest_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    closest_sphere.paint_uniform_color([1.0, 0.65, 0.0])  # orange
    closest_sphere.translate(closest_point)

    template_show3 = copy.deepcopy(aligned_template); template_show3.paint_uniform_color([0, 1, 0])
    scene_show3 = copy.deepcopy(cropped_scene); scene_show3.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries(
        [template_show3, scene_show3, template_sphere, closest_sphere],
        window_name="Template (green) + Scene (red) + Markers"
    )



    # 11) FINAL visualization: ONLY cropped scene + normal line + plane
    scene_only = copy.deepcopy(cropped_scene); scene_only.paint_uniform_color([1, 0, 0])  # red
    o3d.visualization.draw_geometries(
        [scene_only, plane_patch, normal_line],
        window_name="Cropped scene + PCA plane + PCA normal"
    )

    # 12) Export both points (no normals in file unless you want to add it)
    export_points_to_txt(template_point, closest_point)
