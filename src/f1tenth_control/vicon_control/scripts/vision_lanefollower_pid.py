#!/usr/bin/env python3

#================================================================
# File name: vision_lanefollower_pid.py                                                                  
# Description: waypoints tracker using pid                                                                
# Author: Team Thunder [FA24] & PNatarajan123
# Email: hl89@illinois.edu
# Date created: 08/02/2021                                                                
# Date last modified: 03/08/2025                                                          
# Version: 1.0                                                                                                                                         
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
# import os 
# import csv
# import math
# import time
import numpy as np
from numpy import linalg as la
# import scipy.signal as signal

from cv_bridge import CvBridge

# ROS Headers
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from lane_detection import lanenet_detector
from pid_controller import PIDController


class PIDControl(object):
  
    def __init__(self):

        self.rate = rospy.Rate(15)       
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation",
                                            AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"

        # vehicle config
        self.ref_speed = 0.5 # m/s, reference speed
        self.drive_msg.drive.speed = self.ref_speed

        self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # PID controller gains (you can tune these values)
        self.Kp = 0.34
        self.Ki = 0.0   
        self.Kd = 0.24

        # init controllers
        self.lane_detector = lanenet_detector()
        self.controller = PIDController(self.Kp, self.Ki, self.Kd, output_limits=(-0.7, 0.7))

        # Camera parameters (adjust these based on your camera)
        self.image_width = 640  # pixelss
        self.camera_fov_deg = 60  # degrees
        self.radians_per_pixel = (self.camera_fov_deg / self.image_width) * (np.pi / 180)  # radians per pixel

        self.start_time = None


    def carstate_callback(self, carstate_msg):
        self.x   = carstate_msg.data[0]  # meters
        self.y   = carstate_msg.data[1]  # meters
        self.yaw = carstate_msg.data[3]  # degrees


    def start_pid(self):
        # Initialize the previous steering angle
        previous_steering_angle = 0.0

        while not rospy.is_shutdown():
            current_time_ros = rospy.get_time()

            # Get the steering error from the lane detector
            lanenet_steering_error = self.lane_detector.steering_error
            steering_error_radians = lanenet_steering_error * self.radians_per_pixel
            
            # Calculate the weighted steering angle
            f_delta = self.controller.compute(steering_error_radians, current_time_ros)
            weighted_steering_angle = (previous_steering_angle + f_delta)/2 if abs(f_delta-previous_steering_angle) > 0.15 else f_delta

            # Publish the steering command
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = weighted_steering_angle
            self.ctrl_pub.publish(self.drive_msg)

            # Update the previous steering angle
            previous_steering_angle = weighted_steering_angle

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('vicon_pid_node', anonymous=True)
    pid = PIDControl()

    try:
        pid.start_pid()
    except rospy.ROSInterruptException:
        pass
