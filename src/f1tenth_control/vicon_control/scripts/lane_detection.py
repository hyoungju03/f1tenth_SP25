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

        self.lookahead_row = 375
        self.lookahead_col = 0
        self.center_col = 640//2-8
        self.steering_error = 0.0

        self.stop_sign_detected = 0
        self.stop_sign_area = 0

        self.skip_frame = 0

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()

        mask_image, bird_image = self.detection(raw_img)

        # cv2.imwrite('bird_image.png', bird_image)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=50, thresh_max=100):

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
        color_output = self.color_thresh(img)

        # combined_binary = cv2.bitwise_or(sobel_output, color_output)
        combined_binary = color_output

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

    def detect_stop_sign(self, image):
        """
        Detect stop signs in the image using color and shape detection.
        Args:
            image (np.array): The input image (BGR format).
        Returns:
            stop_detected (bool): True if a stop sign is detected on the right side of the screen.
            stop_area (int): The area of the detected stop sign (for debugging or further use).
        """
        stop_detected = False
        stop_area = 0

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the narrow red color range for stop sign detection
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to reduce the number of vertices
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # Check if the contour has 6-10 vertices (octagon-like) and is reasonably large
            if 7 <= len(approx) <= 9 and cv2.contourArea(contour) > 500:
                # Calculate the area of the contour
                stop_area = cv2.contourArea(contour)

                self.stop_sign_area = stop_area

                # Get the left-most vertex
                leftmost_vertex = min(approx, key=lambda point: point[0][0])[0][0]

                # Check if the left-most vertex is on the right side of the screen
                if leftmost_vertex > self.center_col and stop_area > 1400:
                    # Draw the contour and vertices on the image
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                    for point in approx:
                        cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)  # Draw vertices as blue dots

                    stop_detected = True
                    break

        # Save the image with annotations
        cv2.imwrite("cv2_ss_img_with_annotations.png", image)

        return stop_detected, stop_area


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

                # height, width = img.shape[:2]
                # src = np.float32([(140, 300), (40, height), (width-40, height), (width-140, 300)])
                # src = src.astype(int)

                # # Draw the trapezoid
                # for i in range(4):
                #     pt1 = tuple(src[i])
                #     pt2 = tuple(src[(i + 1) % 4])  # Connect to the next point (loop back to start for the last point)
                #     cv2.line(combine_fit_img, pt1, pt2, (0, 255, 0), 2)  # Draw green lines

                # calculate error and draw error on screen
                self.coeff = ret['fit']
                self.lookahead_col = self.coeff[2] + self.coeff[1] * self.lookahead_row + self.coeff[0] * self.lookahead_row**2
                self.steering_error = self.center_col - self.lookahead_col

                # cv2.circle(bird_fit_img, (int(self.lookahead_col), self.lookahead_row), 5, (0,0,255), -1)
                # cv2.circle(bird_fit_img, (self.center_col, self.lookahead_row), 5, (0,0,255), -1)

                # YOLO
                if self.skip_frame > 2:
                    stop_detected, stop_area = self.detect_stop_sign(img)
                    # self.stop_sign_detected = int(stop_detected)
                    self.stop_sign_area = stop_area
                    self.skip_frame = 0

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


# if __name__ == '__main__':
#     # # init args
#     rospy.init_node('lanenet_node', anonymous=True)
#     lanenet_detector()
#     while not rospy.core.is_shutdown():
#         rospy.rostime.wallsleep(0.5)


# def detect_stop_sign(self, image):
#     stop_detected = False
#     stop_area = 0

#     # Convert the image to HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define the red color range for stop sign detection
#     lower_red1 = np.array([0, 70, 50])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 70, 50])
#     upper_red2 = np.array([180, 255, 255])

#     # Create masks for red color
#     mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(mask1, mask2)

#     # Apply morphological operations to clean up the mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Approximate the contour to reduce the number of vertices
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the contour has 8 vertices (octagon) and is reasonably large
#         if len(approx) == 8 and cv2.contourArea(contour) > 500:
#             # Calculate the area of the contour
#             stop_area = cv2.contourArea(contour)

#             # Compute the bounding rectangle and its area
#             x, y, w, h = cv2.boundingRect(approx)
#             rect_area = w * h

#             # Compute the extent (ratio of contour area to bounding rectangle area)
#             extent = float(stop_area) / rect_area

#             # Check if the extent is within a reasonable range (e.g., >0.8)
#             if extent > 0.8:
#                 # Check if sides are approximately equal (regular octagon)
#                 side_lengths = []
#                 for i in range(len(approx)):
#                     pt1 = approx[i][0]
#                     pt2 = approx[(i+1)%len(approx)][0]
#                     side_length = np.linalg.norm(pt1 - pt2)
#                     side_lengths.append(side_length)

#                 # Compute mean and standard deviation of side lengths
#                 mean_side_length = np.mean(side_lengths)
#                 std_side_length = np.std(side_lengths)

#                 # Check if side lengths are approximately equal
#                 if std_side_length / mean_side_length < 0.2:
#                     # Get the left-most vertex
#                     leftmost_vertex = min(approx, key=lambda point: point[0][0])[0][0]

#                     # Check if the left-most vertex is on the right side of the screen
#                     if leftmost_vertex > self.center_col and stop_area > 1400:
#                         # Draw the contour and vertices on the image
#                         cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
#                         for point in approx:
#                             cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)

#                         stop_detected = True
#                         break

#     # Save the image with annotations
#     cv2.imwrite("cv2_ss_img_with_annotations.png", image)

#     return stop_detected, stop_area
