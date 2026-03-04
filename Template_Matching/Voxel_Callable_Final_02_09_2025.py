#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Callable torso point cloud cleanup.

Pipeline:
  1) Voxel downsample
  2) Statistical outlier removal (SOR)
  3) Radius outlier removal (ROR)
  4) Keep largest DBSCAN cluster (torso)
  4.5) Mesh-based signed-distance trim (removes floaters above skin)
  5) Recompute normals

Requirements:
  - Open3D >= 0.16 (for tensor RaycastingScene API)
"""

import os
os.environ.setdefault("O3D_HEADLESS", "1")  # avoid WGL issues on headless/remote
from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d


# ------------------ Default Params (same behavior as your script) ------------------
DEFAULT_PARAMS = dict(
    # Preprocessing
    voxel_size=0.005,             # 5 mm
    stat_nb_neighbors=60,
    stat_std_ratio=1.5,
    radius=0.02,                  # 20 mm
    radius_min_points=20,
    dbscan_eps=0.03,              # 30 mm
    dbscan_min_points=60,
    # SDF trimming
    poisson_depth=6,              # you set 6 in your latest script
    sdf_trim_mm=2.0,              # you set 2.0 mm in your latest script
    use_pre_clean_for_sdf=False,  # filter cleaned torso (safer default)
    snap_to_surface=True,         # project survivors onto mesh (smoothest result)
)
# -------------------------------------------------------------------------------


def _as_pointcloud(pcd_input: Union[str, o3d.geometry.PointCloud, np.ndarray]) -> o3d.geometry.PointCloud:
    """Accept a path (.ply/.pcd/etc), an Open3D PointCloud, or Nx3 numpy array."""
    if isinstance(pcd_input, o3d.geometry.PointCloud):
        if pcd_input.is_empty():
            raise ValueError("Input Open3D point cloud is empty.")
        return pcd_input

    if isinstance(pcd_input, str):
        pcd = o3d.io.read_point_cloud(pcd_input)
        if pcd.is_empty():
            raise ValueError(f"Empty or missing point cloud file: {pcd_input}")
        return pcd

    if isinstance(pcd_input, np.ndarray):
        if pcd_input.ndim != 2 or pcd_input.shape[1] != 3:
            raise ValueError("NumPy array must be of shape (N, 3).")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_input.astype(np.float64))
        return pcd

    raise TypeError("pcd_input must be a file path (str), an Open3D PointCloud, or an (N,3) NumPy array.")


def _estimate_orient_normals(pcd: o3d.geometry.PointCloud, knn: int = 50) -> None:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(knn)


def _show(geoms, title: str, visualize: bool) -> None:
    if not visualize:
        return
    if not isinstance(geoms, list):
        geoms = [geoms]
    o3d.visualization.draw_geometries(geoms, window_name=title)


def _sdf_trim_points(
    pcd_source: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    tau_m: float,
    snap: bool = False,
) -> Tuple[o3d.geometry.PointCloud, int, int]:
    """Keep points whose signed distance <= tau_m (meters)."""
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    pts_np = np.asarray(pcd_source.points)
    if pts_np.size == 0:
        return o3d.geometry.PointCloud(), 0, 0

    pts_t = o3d.core.Tensor(pts_np, dtype=o3d.core.Dtype.Float32)
    sdf_t = scene.compute_signed_distance(pts_t).reshape((-1,))
    sdf_np = np.asarray(sdf_t)  # meters

    # Keep on/under or within small band above the surface
    keep_mask = (sdf_np <= 0.0) | (sdf_np <= tau_m)
    kept_idx = np.where(keep_mask)[0]

    pcd_out = o3d.geometry.PointCloud()
    if kept_idx.size > 0:
        if snap:
            closest = scene.compute_closest_points(pts_t[keep_mask])["points"].numpy()
            pcd_out.points = o3d.utility.Vector3dVector(closest)
        else:
            pcd_out.points = o3d.utility.Vector3dVector(pts_np[kept_idx])

    return pcd_out, int(keep_mask.sum()), int(len(pts_np))


def clean_torso_point_cloud(
    pcd_input: Union[str, o3d.geometry.PointCloud, np.ndarray],
    visualize: bool = False,
    **overrides,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Clean a torso point cloud with your pipeline and return:
      (final_clean_point_cloud, cropped_poisson_mesh)

    Args:
        pcd_input: path to point cloud, Open3D PointCloud, or (N,3) numpy array.
        visualize: if True, shows intermediate steps.
        **overrides: any parameter in DEFAULT_PARAMS can be overridden here.

    Returns:
        pcd_final, mesh_pois
    """
    # Open3D tensor API check

    if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry") or not hasattr(o3d.t.geometry, "RaycastingScene"):
        raise RuntimeError("Open3D >= 0.16 required (missing RaycastingScene).")

    # Merge params
    P = DEFAULT_PARAMS.copy()
    P.update(overrides)

    # 0) Load/convert
    pcd = _as_pointcloud(pcd_input)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # grey
    print(f"[Load] Raw points: {len(pcd.points)}")
    _show(pcd, f"Raw input ({len(pcd.points)} pts)", visualize)

    # 1) Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=P["voxel_size"])
    print(f"[Voxel] {len(pcd_down.points)} pts @ {P['voxel_size']*1000:.1f} mm voxels")
    _estimate_orient_normals(pcd_down, knn=40)
    _show(pcd_down, f"Step 1 — Downsampled ({len(pcd_down.points)} pts)", visualize)

    # 2) Statistical outlier removal
    pcd_stat, ind_stat = pcd_down.remove_statistical_outlier(
        nb_neighbors=P["stat_nb_neighbors"],
        std_ratio=P["stat_std_ratio"]
    )
    removed_sor = len(pcd_down.points) - len(pcd_stat.points)
    print(f"[SOR] {len(pcd_stat.points)} pts (removed {removed_sor})")

    # Optional visualize inliers vs removed
    if visualize:
        outliers_mask = np.ones(len(pcd_down.points), dtype=bool)
        outliers_mask[ind_stat] = False
        outliers_cloud = pcd_down.select_by_index(np.where(outliers_mask)[0])
        inliers_cloud = pcd_stat
        inliers_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        outliers_cloud.paint_uniform_color([1.0, 0.2, 0.2])
        _show([inliers_cloud, outliers_cloud],
              f"Step 2 — After SOR (gray=inliers {len(inliers_cloud.points)}, red=removed {len(outliers_cloud.points)})",
              visualize)

    # 3) Radius outlier removal
    pcd_rad, ind_rad = pcd_stat.remove_radius_outlier(
        nb_points=P["radius_min_points"],
        radius=P["radius"]
    )
    removed_ror = len(pcd_stat.points) - len(pcd_rad.points)
    print(f"[ROR] {len(pcd_rad.points)} pts (removed {removed_ror})")
    _show(pcd_rad, f"Step 3 — After Radius Outlier Removal ({len(pcd_rad.points)} pts)", visualize)

    # 4) DBSCAN keep largest cluster (torso)
    labels = np.array(pcd_rad.cluster_dbscan(
        eps=P["dbscan_eps"], min_points=P["dbscan_min_points"], print_progress=False
    ))
    if labels.size > 0 and labels.max() >= 0:
        sizes = [(labels == i).sum() for i in range(labels.max() + 1)]
        biggest = int(np.argmax(sizes))
        torso = pcd_rad.select_by_index(np.where(labels == biggest)[0])
        rest = pcd_rad.select_by_index(np.where(labels != biggest)[0])
        print(f"[DBSCAN] kept cluster #{biggest} with {len(torso.points)} pts; "
              f"discarded {len(pcd_rad.points)-len(torso.points)} pts")
        if visualize:
            torso.paint_uniform_color([0.5, 0.8, 0.5])
            rest.paint_uniform_color([0.85, 0.5, 0.5])
            _show([torso, rest], f"Step 4 — Cluster Keep (green=kept {len(torso.points)} pts, red=discarded)", visualize)
        pcd_clean_pre_sdf = torso
    else:
        print("[DBSCAN] No clusters found (keeping all points).")
        pcd_clean_pre_sdf = pcd_rad

    # 4.5) Poisson mesh + SDF trim
    print(f"[SDF] Building Poisson mesh (depth={P['poisson_depth']})...")
    # Ensure normals exist for Poisson
    pcd_for_mesh = pcd_clean_pre_sdf

    if not pcd_for_mesh.has_normals():
        _estimate_orient_normals(pcd_for_mesh, knn=50)

    mesh_pois, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_for_mesh, depth=P["poisson_depth"], linear_fit=True
    )
    # Crop away Poisson's infinite sheet
    aabb = pcd_for_mesh.get_axis_aligned_bounding_box()
    aabb = aabb.scale(1.05, aabb.get_center())  # 5% pad
    mesh_pois = mesh_pois.crop(aabb)
    mesh_pois.compute_vertex_normals()
    mesh_pois.paint_uniform_color([0.2, 0.8, 0.5])  # greenish
    _show(mesh_pois, "Step 4.5 — Poisson Mesh (cropped)", visualize)

    # Which cloud to filter?
    pcd_sdf_source = pcd if P["use_pre_clean_for_sdf"] else pcd_clean_pre_sdf
    tau = float(P["sdf_trim_mm"]) / 1000.0  # mm -> m

    pcd_sdf, kept, total = _sdf_trim_points(
        pcd_sdf_source, mesh_pois, tau_m=tau, snap=P["snap_to_surface"]
    )
    print(f"[SDF] kept {kept} / {total} pts with τ={P['sdf_trim_mm']:.1f} mm "
          f"(source={'raw' if P['use_pre_clean_for_sdf'] else 'cleaned torso'})")
    pcd_sdf.paint_uniform_color([0.2, 0.8, 0.5])  # greenish
    _show(pcd_sdf, f"Step 4.5 — After SDF Trim (τ={P['sdf_trim_mm']:.1f} mm)", visualize)

    '''
    # 5) Recompute normals on final
    pcd_final = pcd_sdf
    if len(pcd_final.points) > 0:
        _estimate_orient_normals(pcd_final, knn=50)
    pcd_final.paint_uniform_color([0.2, 0.8, 0.5])  # greenish
    _show(pcd_final, f"Step 5 — Final Cleaned + Normals ({len(pcd_final.points)} pts)", visualize)
    '''
    return pcd_sdf


# ------------------ Optional CLI entrypoint ------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Callable torso point cloud cleanup.")
    ap.add_argument("input", help="Path to input point cloud (PLY/PCD/etc.)")
    ap.add_argument("--visualize", action="store_true", help="Show intermediate visualizations.")
    ap.add_argument("--save", default=None, help="Optional path to save final cleaned point cloud (PLY).")
    # Quick overrides (add more as needed)
    ap.add_argument("--poisson_depth", type=int, default=DEFAULT_PARAMS["poisson_depth"])
    ap.add_argument("--sdf_trim_mm", type=float, default=DEFAULT_PARAMS["sdf_trim_mm"])
    ap.add_argument("--snap_to_surface", action="store_true", default=DEFAULT_PARAMS["snap_to_surface"])
    args = ap.parse_args()

    pcd_final, mesh = clean_torso_point_cloud(
        args.input,
        visualize=args.visualize,
        poisson_depth=args.poisson_depth,
        sdf_trim_mm=args.sdf_trim_mm,
        snap_to_surface=args.snap_to_surface,
    )

    if args.save:
        ok = o3d.io.write_point_cloud(args.save, pcd_final, write_ascii=False, compressed=False)
        print(f"[Save] Wrote {args.save}: {ok}")
