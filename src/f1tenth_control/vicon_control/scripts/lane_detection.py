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
        self.stop_sign_pub=rospy.Publisher("stop sign detected", Bool, queue_size=1)
        self.lane_line = Line(n=5)

        self.detected = False
        self.hist = True
        
        # self.Kp = 0.0
        # self.Ki = 0.0
        # self.Kd = 0.0
        # self.controller = PIDController(self.Kp, self.Ki, self.Kd)

        # self.waypoints = []
        # self.waypoint_gen = WaypointGenerator()

        self.coeff = []

        self.lookahead_row = 400
        self.lookahead_col = 0
        self.center_col = 319
        self.steering_error = 0.0

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

        cv2.imwrite('sign_raw_image.png', raw_img)

        mask_image, bird_image = self.detection(raw_img)

        # cv2.imwrite('raw_image.png', raw_img)
        # cv2.imwrite('bird.png', bird_image)

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
            
            # Create a mask for this range and combine with the existing mask
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
   
        src = np.float32([(70, 300), (50, height), (width-50, height), (width-70, 300)])
        dst = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = np.uint8(cv2.warpPerspective(img, M, (width, height)))

        return warped_img, M, Minv, src, dst


    def calculate_error(self, M_inv):

        self.lookahead_col = self.coeff[2] + self.coeff[1] * self.lookahead_row + self.coeff[0] * self.lookahead_row**2

        # vec = np.array([self.lookahead_col, self.lookahead_row, 1])

        # # print(M_inv)


        # self.lookahead_col, self.lookahead_row, t = M_inv @ vec
        # # print(M_inv @ vec)

        self.steering_error = self.center_col - self.lookahead_col




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
                src_points = np.int32(src)
                cv2.polylines(combine_fit_img, [src_points], isClosed=True, color=(0,0,255), thickness=2)
                if self.skip_frame > 5:
                    inference = self.yolo_model('sign_raw_image.png', verbose=False)[0].boxes
                    stop_sign = False
                    for box in inference:
                        if box.cls[0] == 1:  # Assuming class ID 1 is for the target sign
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(combine_fit_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                            area = np.abs((x2-x1)*(y2-y1))
                            if area >= ... :
                                stop_sign = True
                    self.stop_sign_pub.publish(Bool(data=stop_sign))
                    self.skip_frame = 0

                # calculate error and draw error on screen
                self.coeff = ret['fit']
                self.calculate_error(Minv)

                pts = pts[0]
                pt = pts[np.isclose(pts[:,1],400.1)]
                self.lookahead_col = int(pt[0][0])

                cv2.circle(combine_fit_img, (self.lookahead_col, int(self.lookahead_row)), 5, (0,0,255), -1)
                
                # draw center
                cv2.circle(combine_fit_img, (self.center_col, int(self.lookahead_row)), 5, (255,0,127), -1)

                # self.waypoints = self.waypoint_gen.compute_waypoints(self.coeff)
                # for waypoint in self.waypoints:
                #     cv2.circle(bird_fit_img, waypoint, 5, (65,103,274), -1)

                # newwarp = self.waypoint_gen.waypoint_viz(combine_fit_img, self.coeff, Minv)

                # combine_fit_img = cv2.addWeighted(combine_fit_img, 1, newwarp, 1, 0)

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)