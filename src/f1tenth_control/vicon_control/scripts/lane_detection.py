import time
import math
import numpy as np
import cv2
import rospy
import torch

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

# from pid_controller import PIDController
# from waypoint_generation import WaypointGenerator

class lanenet_detector():
    def __init__(self):
        
        self.bridge = CvBridge()
 
        self.sub_image = rospy.Subscriber('D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.lane_line = Line(n=5)

        self.detected = False
        self.hist = True
        self.coeff = []

        self.lookahead_row = 345
        self.lookahead_col = 0
        self.center_col = 640//2 + 40
        self.steering_error = 0.0

        self.stop_detected = 0
        self.stop_area = 0

        self.go_detected = 0
        self.go_area = 0

        self.skip_frame = 0

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()

        # cv2.imwrite("cv2_ss_image.png", raw_img)

        mask_image, bird_image = self.detection(raw_img)

        cv2.imwrite('bird_image.png', bird_image)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=40, thresh_max=80):

        gs_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gs_image, (5,5), 0)
        
        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        
        sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        binary_output = cv2.inRange(sobel_combined, thresh_min, thresh_max).astype(np.uint8)

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):

        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        ranges = [
            (20, 35, 100, 255, 70, 255),
            (20, 35, 130, 160, 40, 60)
        ]
        
        # Initialize the mask as an empty mask (same size as input)
        yellow_mask = np.zeros(hls_image.shape[:2], dtype=np.uint8)
        
        # Iterate over the ranges and combine masks
        for h_lower, h_upper, l_lower, l_upper, s_lower, s_upper in ranges:
            lower_yellow = np.array([h_lower, l_lower, s_lower])
            upper_yellow = np.array([h_upper, l_upper, s_upper])

            yellow_mask = cv2.bitwise_or(yellow_mask, cv2.inRange(hls_image, lower_yellow, upper_yellow))
        
        return yellow_mask


    def combinedBinaryImage(self, img):

        # sobel_output = self.gradient_thresh(img)
        combined_binary = self.color_thresh(img)

        # combined_binary = cv2.bitwise_or(sobel_output, color_output)
        
        binaryImage = morphology.remove_small_objects(combined_binary.astype('bool'),min_size=50,connectivity=2)
        binaryImage = (binaryImage * 255).astype(np.uint8)

        return binaryImage


    def perspective_transform(self, img, verbose=False):

        height, width = img.shape[:2]
   
        src = np.float32([(35, 300), (35, height), (width-35, height), (width-35, 300)])
        dst = np.float32([[0, 0], [60, height], [width-60, height], [width, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = np.uint8(cv2.warpPerspective(img, M, (width, height)))

        return warped_img, M, Minv, src, dst

    def detect_stop_go(self, image):
        # Initialize detection variables
        stop_detected = False
        stop_area = 0
        go_detected = False
        go_area = 0

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red detection range (for "stop")
        lower_red = np.array([170, 100, 50])
        upper_red = np.array([179, 255, 200])

        # Green detection range (for "go")
        lower_green = np.array([55, 90, 40])
        upper_green = np.array([75, 255, 130])

        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Morphological operations to clean up noise for red
        kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_red)

        # Morphological operations to clean up noise for green
        kernel_green = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_green)

        # Detect stop-like contours (red)
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_red:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            # Arbitrary polygon shape and area thresholds
            if len(approx) > 7:
                if area > 300:
                    # Mark detected stop
                    cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)  # Red contour for stop
                    for point in approx:
                        cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)
                    stop_detected = True
                    stop_area = area
                    break

        # Detect go-like contours (green)
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_green:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            # Arbitrary polygon shape and area thresholds for go
            if len(approx) > 7:
                if area > 300:
                    # Mark detected go
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # Green contour for go
                    for point in approx:
                        cv2.circle(image, tuple(point[0]), 5, (0, 255, 0), -1)
                    go_detected = True
                    go_area = area
                    break

        # Annotate results on the image
        text_offset = 50
        if stop_detected:
            cv2.putText(image, f"STOP DETECTED (Area: {stop_area})", (10, text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            text_offset += 30
        if go_detected:
            cv2.putText(image, f"GO DETECTED (Area: {go_area})", (10, text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imwrite("cv2_ss_img_with_annotations.png", image)

        return stop_detected, stop_area, go_detected, go_area


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv, src, dst = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            if ret is not None:
                fit = ret['fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                lane_inds = ret['lane_inds']
        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)
                if ret is not None:
                    fit = ret['fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    lane_inds = ret['lane_inds']
                    fit = self.lane_line.add_fit(fit)
                    self.detected = True
            else:
                fit = self.lane_line.get_fit()
                ret = tune_fit(img_birdeye, fit)
                if ret is not None:
                    fit = ret['fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    lane_inds = ret['lane_inds']
                    fit = self.lane_line.add_fit(fit)
                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None

            if ret is not None:
                self.skip_frame = self.skip_frame + 1

                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img, pts = final_viz(img, fit, Minv)

                # calculate error and draw error on screen
                self.coeff = ret['fit']
                self.lookahead_col = self.coeff[2] + self.coeff[1] * self.lookahead_row + self.coeff[0] * self.lookahead_row**2
                self.steering_error = self.center_col - self.lookahead_col

                # cv2.circle(bird_fit_img, (int(self.lookahead_col), self.lookahead_row), 5, (0,0,255), -1)
                # cv2.circle(bird_fit_img, (self.center_col, self.lookahead_row), 5, (0,0,255), -1)

                if self.skip_frame > 3:
                    stop_detected, stop_area, go_detected, go_area = self.detect_stop_go(combine_fit_img)
                    self.stop_detected = int(stop_detected)
                    self.stop_area = stop_area
                    self.go_detected = int(go_detected)
                    self.go_area = go_area
                    self.skip_frame = 0

            else:
                print("Unable to detect lanes")

            cv2.imwrite('combine.png', combine_fit_img)

            return combine_fit_img, bird_fit_img
