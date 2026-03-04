#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Callable:
  move_to_p1_with_yaw(p1_out, yaw_xz_deg, ...)

Move to absolute pose at p1_out with TOOL axis aligned to yaw in BASE XZ.
No extra shift.
"""

import math, rospy, actionlib, tf2_ros
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest
from cartesian_control_msgs.msg import FollowCartesianTrajectoryAction, FollowCartesianTrajectoryGoal, CartesianTrajectoryPoint
from geometry_msgs.msg import Pose

# ---------------- lin-algebra helpers ----------------
def vnorm(v): n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) or 1.0; return (v[0]/n, v[1]/n, v[2]/n)
def vcross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vdot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def rot_from_tool_axes(x_tool, y_tool, z_tool): return ((x_tool[0], y_tool[0], z_tool[0]), (x_tool[1], y_tool[1], z_tool[1]), (x_tool[2], y_tool[2], z_tool[2]))
def matmul3(A, B): return tuple(tuple(A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j] for j in range(3)) for i in range(3))
def rot_tool_about_axis(axis, ang_rad):
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    if axis == 'x': return ((1,0,0),(0,c,-s),(0,s,c))
    if axis == 'y': return ((c,0,s),(0,1,0),(-s,0,c))
    if axis == 'z': return ((c,-s,0),(s,c,0),(0,0,1))
    raise ValueError("axis must be x|y|z")
def quat_from_rot(R):
    r00,r01,r02 = R[0]; r10,r11,r12 = R[1]; r20,r21,r22 = R[2]; tr = r00 + r11 + r22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0; qw = 0.25 * S
        qx = (r21 - r12) / S; qy = (r02 - r20) / S; qz = (r10 - r01) / S
    elif (r00 > r11) and (r00 > r22):
        S = math.sqrt(1.0 + r00 - r11 - r22) * 2.0; qw = (r21 - r12) / S
        qx = 0.25 * S; qy = (r01 + r10) / S; qz = (r02 + r20) / S
    elif r11 > r22:
        S = math.sqrt(1.0 + r11 - r00 - r22) * 2.0; qw = (r02 - r20) / S
        qx = (r01 + r10) / S; qy = 0.25 * S; qz = (r12 + r21) / S
    else:
        S = math.sqrt(1.0 + r22 - r00 - r11) * 2.0; qw = (r10 - r01) / S
        qx = (r02 + r20) / S; qy = (r12 + r21) / S; qz = 0.25 * S
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    return (qx/n, qy/n, qz/n, qw/n)

# ---------------- pose builder ----------------
def build_pose_aligning_tool_axis(p_base, yaw_xz_deg,
                                  axis_name="-z", up_base=(0,0,1),
                                  tool_z_offset_m=0.0, use_R_T=False,
                                  roll_offset_deg=180.0, auto_flip_to_up=False):
    yaw = math.radians(yaw_xz_deg)
    ray = vnorm((math.cos(yaw), 0.0, math.sin(yaw)))
    sign = -1.0 if axis_name.startswith('-') else +1.0
    axis = axis_name[1:] if axis_name.startswith('-') else axis_name
    z_up = vnorm(up_base)
    if abs(vdot(z_up, ray)) > 0.98: z_up = vnorm((z_up[0], z_up[1]+1e-3, z_up[2]))
    Xp = ray; Yp = vnorm(vcross(z_up, Xp)); Zp = vnorm(vcross(Xp, Yp))
    if axis == 'x': X_tool=(sign*Xp[0],sign*Xp[1],sign*Xp[2]); Y_tool=Yp; Z_tool=Zp
    elif axis == 'y': Y_tool=(sign*Xp[0],sign*Xp[1],sign*Xp[2]); Z_tool=vnorm(vcross(Y_tool,z_up)); X_tool=vnorm(vcross(Y_tool,Z_tool))
    elif axis == 'z': Z_tool=(sign*Xp[0],sign*Xp[1],sign*Xp[2]); X_tool=vnorm(vcross(z_up,Z_tool)); Y_tool=vnorm(vcross(Z_tool,X_tool))
    else: raise ValueError("axis_name must be x|y|z,-x,-y,-z")
    X_tool=vnorm(X_tool); Y_tool=vnorm(vcross(Z_tool,X_tool)); Z_tool=vnorm(vcross(X_tool,Y_tool))
    R=rot_from_tool_axes(X_tool,Y_tool,Z_tool)
    ang=math.radians(roll_offset_deg)*(sign)
    if abs(roll_offset_deg)>1e-6: R=matmul3(R,rot_tool_about_axis(axis,ang))
    if auto_flip_to_up:
        col={'x':0,'y':1,'z':2}[axis]; aligned=(R[0][col],R[1][col],R[2][col]); da=vdot(up_base,aligned)
        upp=(up_base[0]-da*aligned[0],up_base[1]-da*aligned[1],up_base[2]-da*aligned[2])
        yb=(R[0][1],R[1][1],R[2][1])
        if vdot(upp,yb)<0.0: R=matmul3(R,rot_tool_about_axis(axis,math.pi))
    if use_R_T: R=((R[0][0],R[1][0],R[2][0]),(R[0][1],R[1][1],R[2][1]),(R[0][2],R[1][2],R[2][2]))
    qx,qy,qz,qw=quat_from_rot(R)
    Zb=(R[0][2],R[1][2],R[2][2])
    p_off=(p_base[0]+tool_z_offset_m*Zb[0], p_base[1]+tool_z_offset_m*Zb[1], p_base[2]+tool_z_offset_m*Zb[2])
    pose=Pose(); pose.position.x,pose.position.y,pose.position.z=p_off; pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=qx,qy,qz,qw
    return pose

# ---------------- controller helper ----------------
def switch_to_controller(target):
    switch_srv = rospy.ServiceProxy("controller_manager/switch_controller", SwitchController)
    load_srv   = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
    switch_srv.wait_for_service(5.0); load_srv.wait_for_service(5.0)
    try: load_srv(LoadControllerRequest(name=target))
    except Exception: pass
    JOINTS=["scaled_pos_joint_traj_controller","scaled_vel_joint_traj_controller","pos_joint_traj_controller","vel_joint_traj_controller","forward_joint_traj_controller"]
    CART=["pose_based_cartesian_traj_controller","joint_based_cartesian_traj_controller","forward_cartesian_traj_controller"]
    CONFL=["joint_group_vel_controller","twist_controller"]
    others=JOINTS+CART+CONFL
    if target in others: others.remove(target)
    req=SwitchControllerRequest(stop_controllers=others,start_controllers=[target],strictness=SwitchControllerRequest.BEST_EFFORT,start_asap=True,timeout=2.0)
    switch_srv(req); rospy.loginfo("Switched to %s", target)

# ---------------- action helper ----------------
def send_cartesian_pose(client, pose, duration_s):
    goal=FollowCartesianTrajectoryGoal()
    pt=CartesianTrajectoryPoint(); pt.pose=pose; pt.time_from_start=rospy.Duration(duration_s)
    goal.trajectory.points.append(pt); client.send_goal(goal); client.wait_for_result()
    res=client.get_result(); rospy.loginfo("Action result code: %s", getattr(res,"error_code","n/a"))

# ==================== PUBLIC CALLABLE ====================
def move_to_p1_with_yaw(p1_out, yaw_xz_deg,
                        axis="-z", up_base=(0.0,0.0,1.0),
                        tool_z_offset_m=0.0, roll_offset_deg=180.0,
                        auto_flip_to_up=False, controller="forward_cartesian_traj_controller",
                        abs_move_duration_s=5.0, use_R_T=False):
    """
    Move to absolute pose at p1_out with TOOL axis aligned to yaw_xz_deg.
    """
    try:
        import rospy.core as roscore
        if not roscore.is_initialized():
            rospy.init_node("move_to_p1_with_yaw", anonymous=True)
    except Exception: pass

    switch_to_controller(controller)
    action_name=controller+"/follow_cartesian_trajectory"
    client=actionlib.SimpleActionClient(action_name,FollowCartesianTrajectoryAction)
    rospy.loginfo("Waiting for %s ..." % action_name); client.wait_for_server()

    pose1=build_pose_aligning_tool_axis(p1_out,yaw_xz_deg,
                                        axis_name=axis,up_base=up_base,
                                        tool_z_offset_m=tool_z_offset_m,use_R_T=use_R_T,
                                        roll_offset_deg=roll_offset_deg,auto_flip_to_up=auto_flip_to_up)
    rospy.loginfo("Commanding pose: pos=(%.3f, %.3f, %.3f) quat=(%.4f, %.4f, %.4f, %.4f)",
                  pose1.position.x,pose1.position.y,pose1.position.z,
                  pose1.orientation.x,pose1.orientation.y,pose1.orientation.z,pose1.orientation.w)
    send_cartesian_pose(client,pose1,abs_move_duration_s)
    return {"pose":pose1}

# -------------- optional CLI --------------
if __name__=="__main__":
    out=move_to_p1_with_yaw(
        p1_out=(0.60628265,-0.19247409,0.23412674),
        yaw_xz_deg=112.5
    )
    rospy.loginfo("Done. Final pose: %s", out["pose"])
