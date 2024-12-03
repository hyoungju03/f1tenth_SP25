#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 08/15/2022                                                          
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import time
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

        self.rate = rospy.Rate(5)       
        self.stop_sign = False
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation",
                                            AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.ref_speed = 1.2
        self.drive_msg.drive.speed     = self.ref_speed  # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.start_time = None
        self.stop_duration=5
        self.offset = 0.015  # meters
        self.wheelbase = 0.325  # meters

        self.lane_detector = lanenet_detector()

        # PID controller gains (tune these values)
        self.Kp = 0.6
        self.Ki = 0.0
        self.Kd = 0.3
        self.controller = PIDController(self.Kp, self.Ki, self.Kd, output_limits=(-0.3, 0.3))

        # Camera parameters (adjust these based on your camera)
        self.image_width = 640  # pixels
        self.camera_fov_deg = 60  # degrees
        self.radians_per_pixel = (self.camera_fov_deg / self.image_width) * (np.pi / 180)  # radians per pixel

    def stop_sign_callback(self,msg):
        if msg.data:
            if not self.stop_sign:
                self.stop_sign = True
                self.start_time=time.time()

    def carstate_callback(self, carstate_msg):
        self.x   = carstate_msg.data[0]  # meters
        self.y   = carstate_msg.data[1]  # meters
        self.yaw = carstate_msg.data[3]  # degrees


    def get_f1tenth_state(self):
        # Convert heading to yaw in radians
        curr_yaw = np.radians(self.yaw)

        # Reference point is located at the center of rear axle
        curr_x = self.x - self.offset * np.cos(curr_yaw)
        curr_y = self.y - self.offset * np.sin(curr_yaw)

        return curr_x, curr_y, curr_yaw


    def start_pid(self):
      
        while not rospy.is_shutdown():
            current_time = time.time()

            if self.lane_detector.stop_sign_detected:
                if self.start_time is None:
                    self.start_time = time.time()
                else:
                    elapsed_time = current_time - self.start_time
                    
                    if elapsed_time < self.stop_duration:
                        self.drive_msg.drive.speed = 0.0
                    else:
                        self.stop_sign = False
                        self.drive_msg.drive.speed = self.ref_speed
            else:
                self.drive_msg.drive.speed = self.ref_speed
            
            # curr_x, curr_y, curr_yaw = self.get_f1tenth_state()

            # Get the steering error from the lane detector
            lanenet_steering_error = self.lane_detector.steering_error

            # Convert steering error from pixels to radians
            steering_error_radians = lanenet_steering_error * self.radians_per_pixel

            # Compute the control action using the PID controller
            current_time = rospy.get_time()
            steering_correction = self.controller.compute(steering_error_radians, current_time)

            # Set the steering angle
            # f_delta = np.clip(steering_correction, -0.3, 0.3)
            f_delta = steering_correction

            # Debug statements
            print(f"Steering error (pixels): {lanenet_steering_error}")
            print(f"Steering error (radians): {steering_error_radians}")
            print(f"Applied steering angle (degrees): {f_delta*180/np.pi}")
            print("\n")

            # Publish the steering command
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = f_delta
            self.ctrl_pub.publish(self.drive_msg)
        
            self.rate.sleep()

# def pid_controller():
#     rospy.init_node('vicon_pid_node', anonymous=True)
#     pid = PIDControl()

#     try:
#         pid.start_pid()
#     except rospy.ROSInterruptException:
#         pass

if __name__ == '__main__':
    rospy.init_node('vicon_pid_node', anonymous=True)
    pid = PIDControl()

    try:
        pid.start_pid()
    except rospy.ROSInterruptException:
        pass