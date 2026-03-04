#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline:
  (A) Pre-scout pass to compute targets
    - Move to Point 1 -> start mapper#1 (Function 5)
    - Move to Point 2 -> Point 3 (no dwell)
    - Stop mapper#1 and analyze PLY#1 with Function 1 -> get p1_out / p2_out

  (B) Execution near heart region with continuous recording
    - Move to p1_out (yaw 22.5°, with -5 cm Y shift) -> start mapper#2 (Function 5)
    - Sweep +Y by 0.20 m (Function 3)
    - Move to (x=cur, y=cur, z=p2_out.z) with yaw 22.5° (Function 2)
    - Move to (x=p2_out.x, y=cur, z=p2_out.z) with yaw 67.5° (Function 2)
    - FINAL: Move to actual p2_out pose (Function 2) -> stop mapper#2
"""

import os, sys, signal, subprocess
import rospy
from std_srvs.srv import Trigger

from function_4_joint_moves import move_to_joint_positions
from function_1_scout_from_ply import run_scout_from_ply
from function_2_move_to_pose import move_to_p1_with_yaw
from function_3_sweep_y import move_tcp_along_base_y, lookup_current_pose

# ---------- Waypoints (radians) ----------
POINT1 = [ +0.033284, -2.182733, -0.549152, -1.948856, +1.545188, +0.046578 ]
POINT2 = [ +0.050579, -1.506671, -1.986008, -0.701836, +1.521375, +0.062000 ]
POINT3 = [ +0.090120, -1.625740, -2.643507, +0.607892, +1.463506, +0.061962 ]

# ---------- Motion settings ----------
MOVE_DURATION_S = 5.0
CONTROLLER      = "scaled_pos_joint_traj_controller"  # for Function 4 moves

# ---------- Mapper (Function 5) settings ----------
HERE              = os.path.abspath(os.path.dirname(__file__))
MAPPER_SCRIPT     = os.path.join(HERE, "function_5_lidar_mapper.py")
NODE_NAME         = "lidar_to_base_mapper"
STOP_SRV          = f"/{NODE_NAME}/stop"
PLY_PRE_SCOUT     = os.path.join(HERE, "updated_point_cloud_prescout.ply")   # PLY#1
PLY_EXECUTION     = os.path.join(HERE, "updated_point_cloud_execution.ply")  # PLY#2
MAPPER_DUR_CAP_S  = 300.0  # safety cap; we'll stop explicitly

# ---------- Sweep (Function 3) ----------
SWEEP_DISTANCE_M  = 0.40
SWEEP_DURATION_S  = 3.0
SWEEP_CONTROLLER  = "forward_cartesian_traj_controller"

def stop_mapper_via_service(timeout=15.0):
    rospy.wait_for_service(STOP_SRV, timeout=timeout)
    stop = rospy.ServiceProxy(STOP_SRV, Trigger)
    resp = stop()
    rospy.loginfo("Stop service: success=%s, message='%s'", resp.success, resp.message)
    return resp

def launch_mapper_subprocess(output_ply_path: str):
    """Launch Function 5 writing to the given PLY."""
    return subprocess.Popen(
        [
            sys.executable, MAPPER_SCRIPT,
            f"__name:={NODE_NAME}",
            f"_output_ply:={output_ply_path}",
            f"_duration_sec:={MAPPER_DUR_CAP_S}",
        ],
        cwd=HERE,
    )

def main():
    # Init node
    try:
        import rospy.core as roscore
        if not roscore.is_initialized():
            rospy.init_node("full_pipeline_dual_scans", anonymous=True)
    except Exception:
        pass

    # ---------------- (A) PRE-SCOUT: capture torso & compute targets ----------------
    rospy.loginfo("Moving to Point 1…")
    move_to_joint_positions(POINT1, duration_s=MOVE_DURATION_S, controller=CONTROLLER)

    rospy.loginfo("Starting mapper#1 (pre-scout)…")
    mapper1 = launch_mapper_subprocess(PLY_PRE_SCOUT)

    rospy.loginfo("Moving to Point 2…")
    move_to_joint_positions(POINT2, duration_s=MOVE_DURATION_S, controller=CONTROLLER)

    rospy.loginfo("Moving to Point 3…")
    move_to_joint_positions(POINT3, duration_s=MOVE_DURATION_S, controller=CONTROLLER)

    rospy.loginfo("Stopping mapper#1 via service %s …", STOP_SRV)
    try:
        stop_mapper_via_service(timeout=15.0)
    except Exception as e:
        rospy.logwarn("Stop mapper#1 service failed: %s — sending SIGINT …", e)
        try:
            mapper1.send_signal(signal.SIGINT)
        except Exception:
            pass
    try:
        mapper1.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rospy.logwarn("mapper#1 didn’t exit; sending SIGTERM.")
        mapper1.terminate()
        try:
            mapper1.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rospy.logwarn("Force killing mapper#1.")
            mapper1.kill()

    if not os.path.isfile(PLY_PRE_SCOUT):
        rospy.logerr("Pre-scout PLY not found at %s", PLY_PRE_SCOUT)
        sys.exit(1)

    rospy.loginfo("Analyzing pre-scout PLY: %s", PLY_PRE_SCOUT)
    scout = run_scout_from_ply(ply_path=PLY_PRE_SCOUT, save_json=True, save_fig=True)
    if scout is None:
        rospy.logerr("Scout-from-PLY failed.")
        sys.exit(2)

    p1_out = scout["p1_out"]
    p2_out = scout["p2_out"]
    yaw1   = scout["yaw1_deg"]
    yaw2   = scout["yaw2_deg"]

    # ---------------- (B) EXECUTION near heart with continuous recording ----------------
    # Move to p1_out with your -5 cm Y offset; keep your 22.5° convention (or use yaw1)
    p1_out_shifted = (p1_out[0], p1_out[1] - 0.15, p1_out[2])
    yaw_22_5 = 180 - 22.5  # matches your earlier convention; or use yaw1

    rospy.loginfo("Moving to p1_out (shifted) with yaw %.1f° …", yaw_22_5)
    move_to_p1_with_yaw(p1_out_shifted, yaw_22_5)

    # >>> Start mapper#2 RIGHT AFTER reaching p1_out <<<
    rospy.loginfo("Starting mapper#2 (execution window)…")
    mapper2 = launch_mapper_subprocess(PLY_EXECUTION)

    # Sweep +Y
    rospy.loginfo("Sweeping +Y by %.3f m …", SWEEP_DISTANCE_M)
    move_tcp_along_base_y(
        distance_m=SWEEP_DISTANCE_M,
        base="base",
        ee_hint="tool0_controller",
        duration=SWEEP_DURATION_S,
        controller=SWEEP_CONTROLLER
    )

    # Move to (x=cur, y=cur, z=p2_out.z) with yaw 22.5°
    cur_pose = lookup_current_pose(base="base", ee_hint="tool0_controller")
    cur_x, cur_y = cur_pose.position.x, cur_pose.position.y
    target1_xyz = (cur_x, cur_y, p2_out[2])
    rospy.loginfo("Moving to (x=cur, y=cur, z=p2_out.z=%.3f) with yaw %.1f° …", p2_out[2], yaw_22_5)
    move_to_p1_with_yaw(target1_xyz, yaw_22_5)

    # Move to (x=p2_out.x, y=cur, z=p2_out.z) with yaw 67.5°
    cur_pose2 = lookup_current_pose(base="base", ee_hint="tool0_controller")
    cur_y2 = cur_pose2.position.y
    target2_xyz = (p2_out[0], cur_y2, p2_out[2])
    yaw_67_5 = 180-67.5  # if you need the 180° flip, use (180 - 67.5)
    rospy.loginfo("Moving to (x=p2_out.x=%.3f, y=cur=%.3f, z=p2_out.z=%.3f) with yaw %.1f° …",
                  p2_out[0], cur_y2, p2_out[2], yaw_67_5)
    move_to_p1_with_yaw(target2_xyz, yaw_67_5)

    # FINAL: Move to actual p2_out pose (use scout yaw2)
    rospy.loginfo("Moving to actual p2_out with yaw %.1f° …", yaw2)
    p2_out = scout["p2_out"]  # [x,y,z]
    p2_out_shifted = (p2_out[0], p2_out[1] - 0.15, p2_out[2])
    move_to_p1_with_yaw(p2_out_shifted, yaw_67_5)

    # >>> Stop mapper#2 ONLY AFTER the final move <<<
    rospy.loginfo("Stopping mapper#2 via service %s …", STOP_SRV)
    try:
        stop_mapper_via_service(timeout=15.0)
    except Exception as e:
        rospy.logwarn("Stop mapper#2 service failed: %s — sending SIGINT …", e)
        try:
            mapper2.send_signal(signal.SIGINT)
        except Exception:
            pass
    try:
        mapper2.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rospy.logwarn("mapper#2 didn’t exit; sending SIGTERM.")
        mapper2.terminate()
        try:
            mapper2.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rospy.logwarn("Force killing mapper#2.")
            mapper2.kill()

    if os.path.isfile(PLY_EXECUTION):
        rospy.loginfo("Execution PLY saved: %s", PLY_EXECUTION)
    else:
        rospy.logwarn("Execution PLY not found at %s", PLY_EXECUTION)

    rospy.loginfo("All steps complete.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
