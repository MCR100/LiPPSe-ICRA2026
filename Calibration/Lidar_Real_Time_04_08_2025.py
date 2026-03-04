#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

min_dist = 0.02   # 20 mm
max_dist = 0.50   # 50 cm

def scan_callback(msg):
    angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
    ranges = np.array(msg.ranges)

    mask = np.isfinite(ranges) & (ranges >= min_dist) & (ranges <= max_dist)
    angles_f = angles[mask]
    ranges_f = ranges[mask]

    plt.clf()
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.scatter(angles_f, ranges_f, s=5, c='r')
    ax.set_rmax(max_dist)
    ax.set_rmin(min_dist)
    ax.grid(True)
    ax.set_title("LiDAR Scan (20 mm to 50 cm)", va='bottom')
    plt.pause(0.001)

if __name__ == "__main__":
    rospy.init_node("lidar_polar_plot")
    plt.ion()
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    plt.show()
    rospy.spin()
