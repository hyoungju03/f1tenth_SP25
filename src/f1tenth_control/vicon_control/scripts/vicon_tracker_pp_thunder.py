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
        self.rate = rospy.Rate(10)       
        self.ctrl_pub = rospy.Publisher(
            "/vesc/low_level/ackermann_cmd_mux/input/navigation",
            AckermannDriveStamped, queue_size=1
        )
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.ref_speed = 0.5
        self.drive_msg.drive.speed = self.ref_speed  # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # self.offset = 0.015  # meters
        # self.wheelbase = 0.325  # meters

        self.lane_detector = lanenet_detector()

        # PID controller gains (tune these values)
        self.Kp = 0.325
        self.Ki = 0.0
        self.Kd = 1.35
        # self.Kp = 0.8
        # self.Ki = 0.0
        # self.Kd = 0.0
        self.controller = PIDController(self.Kp, self.Ki, self.Kd, output_limits=(-1, 1))

        # Camera parameters (adjust these based on your camera)
        self.image_width = 640  # pixels
        self.camera_fov_deg = 60  # degrees
        self.radians_per_pixel = (self.camera_fov_deg / self.image_width) * (np.pi / 180)  # radians per pixel

        self.stop_duration = 3.0  # seconds
        self.stopping = False
        self.stop_sign_handled = False
        self.traffic_light = None
        self.red_light_detected = False
        self.start_time = None

    # def stop_sign_callback(self,msg):
    #     if msg.data:
    #         if not self.stop_sign:
    #             self.stop_sign = True
    #             self.start_time = time.time()

    def carstate_callback(self, carstate_msg):
        self.x = carstate_msg.data[0]  # meters
        self.y = carstate_msg.data[1]  # meters
        self.yaw = carstate_msg.data[3]  # degrees

    # def get_f1tenth_state(self):
    #     # Convert heading to yaw in radians
    #     curr_yaw = np.radians(self.yaw)

    #     # Reference point is located at the center of rear axle
    #     curr_x = self.x - self.offset * np.cos(curr_yaw)
    #     curr_y = self.y - self.offset * np.sin(curr_yaw)

    #     return curr_x, curr_y, curr_yaw

    def start_pid(self):
        # Initialize the previous steering angle
        previous_steering_angle = 0.0

        while not rospy.is_shutdown():
            print(f"Stop Sign Detected: {self.lane_detector.stop_sign_detected}, Area: {self.lane_detector.stop_sign_area}")
            print(f"Traffic Light: {self.lane_detector.traffic_light}")

            current_time = time.time()
            self.traffic_light = self.lane_detector.traffic_light

            # Handle red traffic light
            if self.traffic_light == "RED":
                self.red_light_detected = True
                self.drive_msg.drive.speed = 0.0
                print("Red light detected. Stopping...")

            # Handle green traffic light
            elif self.red_light_detected and self.traffic_light == "GREEN":
                self.red_light_detected = False
                self.drive_msg.drive.speed = self.ref_speed
                print("Green light detected. Resuming...")

            # Handle stop sign detection
            elif self.stopping:
                elapsed_time = current_time - self.start_time
                if elapsed_time < self.stop_duration:
                    self.drive_msg.drive.speed = 0.0
                    print("Stopping for stop sign...")
                else:
                    self.drive_msg.drive.speed = self.ref_speed
                    self.stopping = False
                    self.stop_sign_handled = True
                    print("Finished stop sign. Resuming...")
            elif self.lane_detector.stop_sign_detected and not self.stop_sign_handled:
                self.start_time = current_time
                self.stopping = True
                print("Stop sign detected. Initiating stop...")

            # Normal operation when no stop signs or red lights
            elif not self.red_light_detected and not self.stopping:
                self.drive_msg.drive.speed = self.ref_speed

            # Get the steering error from the lane detector
            lanenet_steering_error = self.lane_detector.steering_error
            steering_error_radians = lanenet_steering_error * self.radians_per_pixel

            if abs(steering_error_radians) < 0.06:
                steering_error_radians = 0

            current_time_ros = rospy.get_time()
            steering_correction = self.controller.compute(steering_error_radians, current_time_ros)
            f_delta = steering_correction

            # Calculate the weighted steering angle (50% new, 50% previous)
            weighted_steering_angle = (0.5 * previous_steering_angle) + (0.5 * f_delta)

            # Debug statements
            print(f"Steering error (pixels): {lanenet_steering_error}")
            print(f"Steering error (radians): {steering_error_radians}")
            print(f"Previous steering angle (radians): {previous_steering_angle}")
            print(f"Computed steering angle (radians): {f_delta}")
            print(f"Weighted steering angle (radians): {weighted_steering_angle}")
            print("\n")

            # Update the previous steering angle
            previous_steering_angle = weighted_steering_angle

            # Publish the steering command
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = weighted_steering_angle
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