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

        self.rate = rospy.Rate(15)       
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation",
                                            AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.ref_speed = 0.5
        self.drive_msg.drive.speed = self.ref_speed  # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        self.lane_detector = lanenet_detector()

        # PID controller gains (tune these values)
        self.Kp = 0.34
        self.Ki = 0.0
        self.Kd = 0.24
        # self.Kp = 0.14
        # self.Ki = 0.0
        # self.Kd = 0.11
        # self.Kd = 0.6

        self.controller = PIDController(self.Kp, self.Ki, self.Kd, output_limits=(-0.7, 0.7))

        # Camera parameters (adjust these based on your camera)
        self.image_width = 640  # pixelss
        self.camera_fov_deg = 60  # degrees
        self.radians_per_pixel = (self.camera_fov_deg / self.image_width) * (np.pi / 180)  # radians per pixel

        self.start_time = None
        self.stop_duration = 3.0  # seconds
        self.stopping = False
        self.stop_sign_handled = False

        # Variables for the scenarios
        self.first_stop_done_time = None
        self.second_scenario = False

        # For first scenario delay
        self.stop_sign_detect_time = None
        self.wait_before_stopping = False


    def carstate_callback(self, carstate_msg):
        self.x   = carstate_msg.data[0]  # meters
        self.y   = carstate_msg.data[1]  # meters
        self.yaw = carstate_msg.data[3]  # degrees


    def reset_scenarios(self):
        # Reset variables so that the logic can be repeated again.
        self.stop_sign_handled = False
        self.first_stop_done_time = None
        self.second_scenario = False
        self.wait_before_stopping = False
        self.stopping = False
        print("Scenarios reset. Ready for another loop.")


    def start_pid(self):
        # Initialize the previous steering angle
        previous_steering_angle = 0.0

        while not rospy.is_shutdown():
            print(self.lane_detector.stop_detected, self.lane_detector.stop_area)

            current_time = time.time()

            # Stop sign: stop for 3 seconds, then go.
            # Wait 1 second after detection before actually stopping.

            # Red light: stop and wait until 'go' is detected.
            # After handling the stop light, reset everything so the cycle can repeat.

            if self.wait_before_stopping:
                # If we detected a stop sign but we are waiting 1 second before stopping
                if current_time - self.stop_sign_detect_time >= 1.0:
                    # Now proceed to stop
                    self.start_time = current_time
                    self.stopping = True
                    self.wait_before_stopping = False
                    print("Scenario 1: 1 second passed after detection. Now stopping for 3 seconds...")
                else:
                    # Keep driving during the 1 second wait
                    self.drive_msg.drive.speed = self.ref_speed

            elif self.stopping and not self.second_scenario:
                # Handling first scenario (stop sign)
                elapsed_time = current_time - self.start_time
                if elapsed_time < self.stop_duration:
                    self.drive_msg.drive.speed = 0.0
                    print("Scenario 1: Stopped at stop sign.")
                else:
                    self.drive_msg.drive.speed = self.ref_speed
                    self.stopping = False
                    self.stop_sign_handled = True
                    self.first_stop_done_time = time.time()
                    print("Scenario 1: 3 seconds passed. Proceeding after stop sign.")
            elif self.stopping and self.second_scenario:
                # Handling second scenario (red light)
                if self.lane_detector.go_detected:
                    self.drive_msg.drive.speed = self.ref_speed
                    self.stopping = False
                    self.second_scenario = False
                    print("Scenario 2: Green light detected, proceeding.")
                    # After handling the second scenario, reset so it can happen again
                    self.reset_scenarios()
                else:
                    self.drive_msg.drive.speed = 0.0
                    print("Scenario 2: Red light detected. Stopped and waiting for green light.")
            elif self.lane_detector.stop_detected and not self.stop_sign_handled and not self.wait_before_stopping:
                # Detected first stop sign scenario
                # Wait 1 second before actually stopping
                self.stop_sign_detect_time = current_time
                self.wait_before_stopping = True
                print("Scenario 1: Stop sign detected. Waiting 1 second before stopping...")
            elif self.lane_detector.stop_detected and self.stop_sign_handled and self.first_stop_done_time is not None:
                # Check if at least 5 seconds have passed since first scenario ended
                if (time.time() - self.first_stop_done_time) > 5.0:
                    # Second scenario: red light detected
                    self.stopping = True
                    self.second_scenario = True
                    print("Scenario 2: Red light detected. Waiting until green light is detected...")
            else:
                # Resume normal operation if nothing special is happening
                if not self.stopping and not self.wait_before_stopping:
                    self.drive_msg.drive.speed = self.ref_speed

            # Get the steering error from the lane detector
            lanenet_steering_error = self.lane_detector.steering_error
            steering_error_radians = lanenet_steering_error * self.radians_per_pixel

            # if abs(steering_error_radians) < 0.02:
            #     steering_error_radians = 0

            current_time_ros = rospy.get_time()
            steering_correction = self.controller.compute(steering_error_radians, current_time_ros)
            f_delta = steering_correction

            # Calculate the weighted steering angle (60% new, 40% previous)
            # weighted_steering_angle = (0.5 * previous_steering_angle) + (0.5 * f_delta)
            weighted_steering_angle = (previous_steering_angle + f_delta)/2 if abs(f_delta-previous_steering_angle) > 0.15 else f_delta
            # weighted_steering_angle = f_delta

            # Debug statements
            print(f"Steering error (pixels): {lanenet_steering_error}")
            print(f"Steering error (radians): {steering_error_radians}")
            print(f"Previous steering angle (radians): {previous_steering_angle}")
            print(f"Computed steering angle (radians): {f_delta}")
            print(f"Weighted steering angle (degree): {weighted_steering_angle/np.pi*180}")
            print("\n")

            # Update the previous steering angle
            previous_steering_angle = weighted_steering_angle

            # Publish the steering command
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = weighted_steering_angle
            self.ctrl_pub.publish(self.drive_msg)

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('vicon_pid_node', anonymous=True)
    pid = PIDControl()

    try:
        pid.start_pid()
    except rospy.ROSInterruptException:
        pass
