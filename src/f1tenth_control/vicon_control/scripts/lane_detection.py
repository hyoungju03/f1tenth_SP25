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

from ultralytics import YOLO

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

        self.lookahead_row = 400
        self.lookahead_col = 0
        self.center_col = 640//2
        self.steering_error = 0.0

        self.stop_sign_detected = 0

        self.skip_frame = 0
        # Load YOLO model for sign detection
        self.yolo_model = YOLO('best.pt')

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):

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

        sobel_output = self.gradient_thresh(img)
        color_output = self.color_thresh(img)

        combined_binary = cv2.bitwise_or(sobel_output, color_output)

        binaryImage = morphology.remove_small_objects(combined_binary.astype('bool'),min_size=50,connectivity=2)
        binaryImage = (binaryImage * 255).astype(np.uint8)

        return binaryImage


    def perspective_transform(self, img, verbose=False):

        height, width = img.shape[:2]
   
        src = np.float32([(130, 300), (35, height), (width-35, height), (width-130, 300)])
        dst = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = np.uint8(cv2.warpPerspective(img, M, (width, height)))

        return warped_img, M, Minv, src, dst
    

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
                # if self.skip_frame > 5:
                #     inference = self.yolo_model('sign_raw_image.png', verbose=False)[0].boxes
                #     # stop_sign = False
                #     if inference is None:
                #         self.stop_sign_detected = 0
                #     else:
                #         for box in inference:
                #             if box.cls[0] == 1:  # Assuming class ID 1 is for the target sign
                #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                #                 # cv2.rectangle(combine_fit_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                #                 area = np.abs((x2-x1)*(y2-y1))
                #                 if area >= 800:
                #                     self.stop_sign_detected = 1
                #             else:
                #                 self.stop_sign_detected = 0
                #     self.skip_frame = 0

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)