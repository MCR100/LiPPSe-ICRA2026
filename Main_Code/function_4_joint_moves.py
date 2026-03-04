#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Callable: move_to_joint_positions(joint_positions, ...)

Moves the arm to an absolute joint configuration (in radians) using a joint trajectory controller.
- Switches controllers the same way as your other scripts.
- Uses FollowJointTrajectoryAction.
- Keeps the API minimal and callable from other scripts.

Example:
    target = [0.033272, -2.182725, -0.549147, -1.948866, 1.545174, 0.046592]
    move_to_joint_positions(target, duration_s=5.0)
"""

import sys
import math
import rospy
import actionlib
from controller_manager_msgs.srv import (
    SwitchController, SwitchControllerRequest,
    LoadController, LoadControllerRequest
)
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

# --- Joint names (same order as your previous scripts) ---
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# --- Available controllers (consistent with your setup) ---
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",     # default we’ll use
]

CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]

CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]


# ---------------- Controller switching (same style as before) ----------------
def switch_to_controller(target_controller: str):
    switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
    load_srv   = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
    switch_srv.wait_for_service(5.0)
    load_srv.wait_for_service(5.0)

    # Load target (ignore error if already loaded)
    try:
        load_srv(LoadControllerRequest(name=target_controller))
    except Exception:
        pass

    others = JOINT_TRAJECTORY_CONTROLLERS + CARTESIAN_TRAJECTORY_CONTROLLERS + CONFLICTING_CONTROLLERS
    if target_controller in others:
        others.remove(target_controller)

    req = SwitchControllerRequest(
        stop_controllers=others,
        start_controllers=[target_controller],
        strictness=SwitchControllerRequest.BEST_EFFORT,
        start_asap=True,
        timeout=2.0,
    )
    switch_srv(req)
    rospy.loginfo("Switched to %s", target_controller)


# ----------------------- PUBLIC CALLABLE -----------------------
def move_to_joint_positions(
    joint_positions,
    duration_s: float = 5.0,
    controller: str = "forward_joint_traj_controller",
    init_node_name: str = "move_to_joint_positions_node"
):
    """
    Move to an absolute joint configuration (radians).

    Args:
        joint_positions: iterable of 6 floats (radians) in the order:
            [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
             wrist_1_joint, wrist_2_joint, wrist_3_joint]
        duration_s: total trajectory time (seconds)
        controller: which joint trajectory controller to use
        init_node_name: ROS node name if we initialize here

    Returns:
        action result object (or None if unavailable)
    """
    # Ensure ROS node is initialized (safe if already initialized)
    try:
        import rospy.core as roscore
        if not roscore.is_initialized():
            rospy.init_node(init_node_name, anonymous=True)
    except Exception:
        pass

    # Validate input length
    if len(joint_positions) != 6:
        raise ValueError("joint_positions must have 6 elements (radians). Got %d" % len(joint_positions))

    # Switch controller
    switch_to_controller(controller)

    # Create action client
    action_name = controller + "/follow_joint_trajectory"
    client = actionlib.SimpleActionClient(action_name, FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for %s ..." % action_name)
    client.wait_for_server()

    # Build trajectory
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions = list(float(x) for x in joint_positions)
    pt.time_from_start = rospy.Duration(float(duration_s))
    goal.trajectory.points.append(pt)

    rospy.loginfo(
        "Sending joint goal (duration=%.2f s):\n  %s",
        duration_s,
        ", ".join(f"{n}={v:.6f} rad" for n, v in zip(JOINT_NAMES, pt.positions))
    )

    # Send and wait
    client.send_goal(goal)
    client.wait_for_result()
    res = client.get_result()
    rospy.loginfo("Trajectory execution finished with result: %s", getattr(res, "error_code", "n/a"))
    return res


# ----------------------- Demo when run directly -----------------------
if __name__ == "__main__":
    try:
        # Example using your provided radians
        target = [0.033272, -2.182725, -0.549147, -1.948866, 1.545174, 0.046592]
        move_to_joint_positions(target, duration_s=5.0, controller="forward_joint_traj_controller")
    except rospy.ROSInterruptException:
        pass