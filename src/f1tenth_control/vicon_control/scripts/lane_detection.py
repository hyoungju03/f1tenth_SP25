import time
import math
import numpy as np
import cv2
import rospy
import torch
from skimage import morphology

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32


class lanenet_detector():
    def __init__(self):
        
        self.bridge = CvBridge()
    
        # ROS publishers / subscribers
        self.sub_image = rospy.Subscriber('D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        # lane detection config
        self.lane_line = Line(n=5)
        self.detected = False
        self.hist = True
        
        # stores coefficients of fit quadratic polynomial
        self.coeff = []

        # feel free to experiment with these values for lane following
        self.lookahead_row = 345
        self.lookahead_col = 0
        self.center_col = 640//2 + 40
        self.steering_error = 0.0

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


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Filters out yellow pixels from image.
        Depending on the lighting condition, the static color mask may not work well.
        It could be helpful to keep track of lighting condition that works best with your car and codebase,
        or you can modify this value.
        """
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


    def getBinaryImage(self, img):
        """
        Post-processing the result from color_thresh to remove noise
        """
        binaryImage = self.color_thresh(img)

        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        binaryImage = (binaryImage * 255).astype(np.uint8)

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Bird's eye view transformation of the image
        """
        height, width = img.shape[:2]

        src = np.float32([(35, 300), (35, height), (width-35, height), (width-35, 300)])
        dst = np.float32([[0, 0], [60, height], [width-60, height], [width, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = np.uint8(cv2.warpPerspective(img, M, (width, height)))

        return warped_img, M, Minv, src, dst


    def detection(self, img):

        binary_img = self.getBinaryImage(img)
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

                # calculate steering error
                self.coeff = ret['fit']
                self.lookahead_col = self.coeff[2] + self.coeff[1] * self.lookahead_row + self.coeff[0] * self.lookahead_row**2
                self.steering_error = self.center_col - self.lookahead_col

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img
