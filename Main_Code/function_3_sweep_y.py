#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Move the current Cartesian TCP by a relative offset along BASE +Y.
Keeps the current orientation.

Defaults (from your TF tree):
  base = 'base'
  ee   = 'tool0_controller'   # << TF exists: base -> tool0_controller

Params you can override:
  _base:=base
  _ee:=tool0_controller
  _distance_m:=0.15
  _duration_sec:=3.0
  _controller:=forward_cartesian_traj_controller
"""

import rospy, actionlib, tf2_ros
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest
from cartesian_control_msgs.msg import FollowCartesianTrajectoryAction, FollowCartesianTrajectoryGoal, CartesianTrajectoryPoint
from geometry_msgs.msg import Pose

# ---------------- Controller helpers ----------------
def switch_to_controller(target):
    switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
    load_srv   = rospy.ServiceProxy("controller_manager/load_controller",   LoadController)
    switch_srv.wait_for_service(5.0); load_srv.wait_for_service(5.0)
    try:
        load_srv(LoadControllerRequest(name=target))
    except Exception:
        pass
    JOINTS = ["scaled_pos_joint_traj_controller","scaled_vel_joint_traj_controller",
              "pos_joint_traj_controller","vel_joint_traj_controller","forward_joint_traj_controller"]
    CART   = ["pose_based_cartesian_traj_controller","joint_based_cartesian_traj_controller",
              "forward_cartesian_traj_controller"]
    CONFL  = ["joint_group_vel_controller","twist_controller"]
    others = JOINTS + CART + CONFL
    if target in others: others.remove(target)
    req = SwitchControllerRequest(stop_controllers=others, start_controllers=[target],
                                  strictness=SwitchControllerRequest.BEST_EFFORT, start_asap=True, timeout=2.0)
    switch_srv(req)
    rospy.loginfo("Switched to %s", target)

# ---------------- TF helpers ----------------
def lookup_current_pose(base, ee_hint=None):
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    lst = tf2_ros.TransformListener(buf)
    rospy.sleep(0.2)

    # Prefer the hint, else try frames that actually exist in your tree
    candidates = []
    if ee_hint:
        candidates.append(ee_hint)
    candidates += ["tool0_controller", "tool0", "flange", "wrist_3_link", "ee_link", "tcp", "end_effector", "link6"]

    last_err = None
    for ee in candidates:
        try:
            tr = buf.lookup_transform(base, ee, rospy.Time(0), rospy.Duration(1.0))
            p = Pose()
            p.position.x = tr.transform.translation.x
            p.position.y = tr.transform.translation.y
            p.position.z = tr.transform.translation.z
            p.orientation = tr.transform.rotation
            rospy.loginfo("Using EE frame: '%s'", ee)
            return p
        except Exception as e:
            last_err = e

    raise RuntimeError("Could not find an EE frame relative to base='%s'. "
                       "Tried: %s. Last TF error: %s" %
                       (base, ", ".join(candidates), last_err))

# ---------------- ROS init helper (new, tiny) ----------------
def _init_node_if_needed(name):
    try:
        if not rospy.core.is_initialized():
            rospy.init_node(name)
    except Exception:
        pass

# ---------------- Callable function (new) ----------------
def move_tcp_along_base_y(distance_m,
                          base="base",
                          ee_hint="tool0_controller",
                          duration=3.0,
                          controller="forward_cartesian_traj_controller"):
    """
    Callable API: moves TCP along BASE +Y by distance_m (meters), keeping orientation.
    Returns the action result (or None if unavailable).
    """
    _init_node_if_needed("move_tcp_along_base_y")

    switch_to_controller(controller)

    cur = lookup_current_pose(base, ee_hint)
    rospy.loginfo("Current pose: x=%.3f y=%.3f z=%.3f  quat=(%.4f,%.4f,%.4f,%.4f)",
                  cur.position.x, cur.position.y, cur.position.z,
                  cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w)

    # Relative move along BASE +Y
    tgt = Pose()
    tgt.position.x = cur.position.x
    tgt.position.y = cur.position.y + float(distance_m)
    tgt.position.z = cur.position.z
    tgt.orientation = cur.orientation  # keep orientation

    rospy.loginfo("Target pose:  x=%.3f y=%.3f z=%.3f  (Δy=%.3f m, duration=%.2f s)",
                  tgt.position.x, tgt.position.y, tgt.position.z, float(distance_m), float(duration))

    action_name = controller + "/follow_cartesian_trajectory"
    client = actionlib.SimpleActionClient(action_name, FollowCartesianTrajectoryAction)
    rospy.loginfo("Waiting for %s ..." % action_name)
    client.wait_for_server()

    goal = FollowCartesianTrajectoryGoal()
    pt = CartesianTrajectoryPoint()
    pt.pose = tgt
    pt.time_from_start = rospy.Duration(float(duration))
    goal.trajectory.points.append(pt)

    client.send_goal(goal)
    client.wait_for_result()
    res = client.get_result()
    rospy.loginfo("Result code: %s", getattr(res, "error_code", "n/a"))
    return res

# ---------------- Main ----------------
def main():
    rospy.init_node("move_tcp_along_base_y")

    base       = rospy.get_param("~base", "base")
    ee_hint    = rospy.get_param("~ee",   "tool0_controller")  # << default fixed
    distance_m = float(rospy.get_param("~distance_m", 0.15))
    duration   = float(rospy.get_param("~duration_sec", 3.0))
    controller = rospy.get_param("~controller", "forward_cartesian_traj_controller")

    move_tcp_along_base_y(distance_m, base=base, ee_hint=ee_hint, duration=duration, controller=controller)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
