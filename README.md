# LiPPSe
LiPPSe: LiDAR-based Probe Pose Seeding for Teleoperated Echocardiography.

## Quick Start
Use this if you want the shortest path to run:

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```
2. Start ROS master and your robot drivers/controllers.
3. Verify LiDAR and TF are available (`/scan`, `base`, `tool0` or your configured EE frame).
4. If hardware mount changed, run calibration:
```bash
python Calibration/Calibration_THE_END_09_2025.py
```
5. Run the main LiPPSe pipeline:
```bash
python run_main_pipeline.py
```
6. (Optional) Run template matching/post-processing:
```bash
python Template_Matching/TEMPLATE_MATCHING_THE_END_02_09_2025.py
```
## 1) Project Purpose
This repository contains a full pipeline to:
1. Calibrate a 2D LiDAR mounted near the robot TCP.
2. Scan and reconstruct torso-region point clouds in the robot base frame.
3. Compute seed probe poses from geometry (`p1_out`, `p2_out`) for robot motion.
4. Optionally run template matching and local normal estimation for correspondence-based targeting.

The code is organized into three folders:
- `Calibration/`: extrinsic LiDAR-to-TCP calibration and quick scan visualization.
- `Main_Code/`: online robot motion + LiDAR scanning + geometric seed pose generation.
- `Template_Matching/`: torso cloud cleanup and template-to-scene registration workflow.

## 2) Recommended Workflow Order
Use this order unless you intentionally run a subset:

1. `Calibration` (only when needed).
2. `Main_Code` (core acquisition + motion pipeline).
3. `Template_Matching` (post-processing / correspondence / local PCA normal).

### When calibration is needed
Run calibration when:
- the LiDAR mounting changed,
- TCP/tooling changed,
- robot-sensor rig was reassembled,
- scan alignment quality degrades.

You can skip calibration when your mount is unchanged and you trust the current `T_TCP_LIDAR` used by the mapper.

## 3) Repository Structure
```text
LiPPSe-ICRA2026/
  Calibration/
    Calibration_THE_END_09_2025.py
    Lidar_Real_Time_04_08_2025.py
  Main_Code/
    main_pipeline.py
    function_1_scout_from_ply.py
    function_2_move_to_pose.py
    function_3_sweep_y.py
    function_4_joint_moves.py
    function_5_lidar_mapper.py
  Template_Matching/
    TEMPLATE_MATCHING_THE_END_02_09_2025.py
    Voxel_Callable_Final_02_09_2025.py
    template_colored_1.ply
  run_main_pipeline.py
```

## 4) Main Pipeline (`Main_Code`) - What Runs
Primary entrypoint:
- `run_main_pipeline.py` (repo root) -> calls `Main_Code/main_pipeline.py`.

`main_pipeline.py` sequence:
1. Move robot through predefined joint waypoints (`POINT1`, `POINT2`, `POINT3`) via `function_4_joint_moves.py`.
2. Launch mapper (`function_5_lidar_mapper.py`) and collect pre-scout cloud:
   - output: `Main_Code/updated_point_cloud_prescout.ply`.
3. Analyze pre-scout cloud with `function_1_scout_from_ply.py` to compute:
   - `p1_out`, `p2_out`, and ray-based yaw angles.
4. Move to `p1_out` (with configured Y shift), start mapper again for execution scan.
5. Perform sweep and intermediate Cartesian moves:
   - `function_3_sweep_y.py` and `function_2_move_to_pose.py`.
6. Final move near `p2_out`, stop mapper service.
7. Save execution cloud:
   - `Main_Code/updated_point_cloud_execution.ply`.

### Internal module roles
- `function_5_lidar_mapper.py`: converts `LaserScan` to 3D base-frame points using TF and fixed `T_TCP_LIDAR`; writes ASCII PLY.
- `function_1_scout_from_ply.py`: torso cleanup (grid component on XZ), ray geometry (22.5 deg / 67.5 deg), pose extraction, optional JSON/PNG.
- `function_2_move_to_pose.py`: absolute Cartesian pose command with yaw-based orientation construction.
- `function_3_sweep_y.py`: relative Cartesian move along base +Y.
- `function_4_joint_moves.py`: absolute joint-space waypoint execution.

## 5) Calibration Folder - What It Does
- `Calibration_THE_END_09_2025.py`:
  - Collects multiple robot poses.
  - Extracts LiDAR sector points and fits line inliers (RANSAC).
  - Solves extrinsic transform using robust least squares.
  - Intended output: calibrated `T_TCP_LIDAR` to be copied into mapper.
- `Lidar_Real_Time_04_08_2025.py`:
  - Real-time polar plot sanity check for `/scan`.

Important calibration-related params (script defaults):
- topic: `/scan`
- frames: `base` -> `tool0`
- sector: `135` to `225` degrees
- range limits: `0.05` to `8.0` m (calibration script)

## 6) Template Matching Folder - What It Does
Pipeline in `TEMPLATE_MATCHING_THE_END_02_09_2025.py`:
1. Clean input scene cloud via `clean_torso_point_cloud(...)`.
2. Load colored template (`template_colored_1.ply`) with a white marker point.
3. Perform feature-based global registration + ICP refinement.
4. Locate white marker on aligned template.
5. Find nearest point in cropped scene.
6. Compute local PCA normal and visualize local plane.
7. Export matched points to text.

Cleanup module `Voxel_Callable_Final_02_09_2025.py` performs:
- voxel downsample,
- SOR + ROR outlier removal,
- DBSCAN largest-cluster keep,
- Poisson + signed-distance trim.

## 7) Prerequisites
This code assumes a ROS1 robot setup with:
- running ROS master and robot drivers,
- valid TF tree including `base` and tool frame (commonly `tool0` / `tool0_controller`),
- available controllers used by scripts (joint + cartesian trajectory controllers),
- LiDAR publishing `sensor_msgs/LaserScan` on `/scan`.

Python libraries used across scripts:
- `numpy`, `scipy`, `matplotlib`
- `open3d` (template matching / cleanup)
- ROS Python stack (`rospy`, message/service packages)
- Install pip packages with `requirements.txt`; install ROS packages via your ROS distribution (not pip).

## 8) How To Run
### Core end-to-end motion + scan pipeline
From repository root:
```bash
python run_main_pipeline.py
```

### Calibration (when needed)
```bash
python Calibration/Calibration_THE_END_09_2025.py
```

### Template matching stage
```bash
python Template_Matching/TEMPLATE_MATCHING_THE_END_02_09_2025.py
```

## 9) Key Outputs
- `Main_Code/updated_point_cloud_prescout.ply`
- `Main_Code/updated_point_cloud_execution.ply`
- scout JSON/PNG generated by `function_1_scout_from_ply.py` (path depends on runtime args/defaults)
- `points.txt` from template matching correspondence export

## 10) Notes for Reproducibility
- `Template_Matching/TEMPLATE_MATCHING_THE_END_02_09_2025.py` currently contains a hardcoded `PLY_IN` example path; update it before running on a new machine.
- Keep frame names and controller names consistent with your robot configuration.
- If calibration is updated, propagate the new `T_TCP_LIDAR` into `Main_Code/function_5_lidar_mapper.py`.




