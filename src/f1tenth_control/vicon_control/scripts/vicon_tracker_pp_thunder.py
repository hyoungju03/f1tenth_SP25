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


class PurePursuit(object):
    
    def __init__(self):
        
        # 0.5 - 0.1 - 0.41

        self.rate = rospy.Rate(30)

        self.look_ahead = 0.3 # 4
        self.wheelbase  = 0.325 # meters
        self.offset     = 0.015 # meters        
        
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.drive_msg.drive.speed     = 1.2 # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        self.lane_detector = lanenet_detector()

        # read waypoints into the system 
        self.goal = 0            
        self.read_waypoints() 
        
    def carstate_callback(self, carstate_msg):
        self.x   = carstate_msg.data[0] # meters
        self.y   = carstate_msg.data[1] # meters
        self.yaw = carstate_msg.data[3] # degrees

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/xyhead_demo_pp.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_x_record   = [float(point[0]) for point in path_points] # x
        self.path_points_y_record   = [float(point[1]) for point in path_points] # y
        self.path_points_yaw_record = [float(point[2]) for point in path_points] # yaw
        self.wp_size                = len(self.path_points_x_record)
        self.dist_arr               = np.zeros(self.wp_size)

    def get_f1tenth_state(self):

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = np.radians(self.yaw)

        # reference point is located at the center of rear axle
        curr_x = self.x - self.offset * np.cos(curr_yaw)
        curr_y = self.y - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
        
        while not rospy.is_shutdown():

            self.path_points_x = np.array(self.path_points_x_record)
            self.path_points_y = np.array(self.path_points_y_record)

            curr_x, curr_y, curr_yaw = self.get_f1tenth_state()

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.05) & (self.dist_arr > self.look_ahead - 0.05) )[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                # find correct look-ahead point by using heading information
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    break

            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # find the curvature and the angle 
            alpha = np.radians(self.path_points_yaw_record[self.goal]) - curr_yaw

            # ----------------- tuning this part as needed -----------------
            k       = 0.12
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.3, 0.3), 3)

            f_delta_deg = round(np.degrees(f_delta))

            print("Current index: " + str(self.goal))
            ct_error = round(np.sin(alpha) * L, 3)
            print("Crosstrack Error: " + str(ct_error))
            print("Front steering angle: " + str(f_delta_deg) + " degrees")
            print("\n")

            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = f_delta
            self.ctrl_pub.publish(self.drive_msg)
        
            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('vicon_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


class PIDControl(object):
  
    def __init__(self):

        self.rate = rospy.Rate(30)       
        self.stop_sign = False
        #    self.stop_sign_sub = rospy.Subscriber("stop sign detected",Bool,self.stop_sign_callback)
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation",
                                            AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.drive_msg.drive.speed     = 1.2  # m/s, reference speed

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
        self.Kp = 0.85
        self.Ki = 0.0
        self.Kd = 0.0
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
                        self.drive_msg.drive.speed = 1.2
            else:
                self.drive_msg.drive.speed = 1.2
            
            curr_x, curr_y, curr_yaw = self.get_f1tenth_state()

            # Get the steering error from the lane detector
            lanenet_steering_error = self.lane_detector.steering_error

            # Convert steering error from pixels to radians
            steering_error_radians = lanenet_steering_error * self.radians_per_pixel

            # Compute the control action using the PID controller
            current_time = rospy.get_time()
            steering_correction = self.controller.compute(steering_error_radians, current_time)

            # Set the steering angle
            f_delta = np.clip(steering_correction, -0.3, 0.3)

            # Debug statements
            print(f"Steering error (pixels): {lanenet_steering_error}")
            print(f"Steering error (radians): {steering_error_radians}")
            print(f"Steering correction (degrees): {steering_correction*180/np.pi}")
            print(f"Applied steering angle (degrees): {f_delta*180/np.pi}")
            print("\n")

            # Publish the steering command
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = f_delta
            self.ctrl_pub.publish(self.drive_msg)
        
            self.rate.sleep()

def pid_controller():
    rospy.init_node('vicon_pid_node', anonymous=True)
    pid = PIDControl()

    try:
        pid.start_pid()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    pid_controller()