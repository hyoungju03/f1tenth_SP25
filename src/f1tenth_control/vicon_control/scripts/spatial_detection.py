import time
import math
import numpy as np
import cv2
import rospy
import torch

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from ultralytics import YOLO
import torch

class spatial_detector():
    def __init__(self):
      
        self.bridge = CvBridge()
        
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.sub_image = rospy.Subscriber('D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_lane_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.lane_line = Line(n=5)

        self.detected = False
        self.hist = True
        
        # Load YOLO model for sign detection
        self.yolo_model = YOLO('best.pt').to(torch.device('cuda'))
        self.skip_frame = 0

    def img_callback(self, data):
      
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Make a copy of the original image for processing
        raw_img = cv_image.copy()

        cv2.imwrite('sign_raw_image.png', raw_img)

        # Perform lane and sign detection
        lane_mask, bird_image = self.detection(raw_img)

        # Publish lane detection images if detected
        if lane_mask is not None:
            out_lane_msg = self.bridge.cv2_to_imgmsg(lane_mask, 'bgr8')
            self.pub_lane_image.publish(out_lane_msg)
        
        if bird_image is not None:
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
            self.pub_bird.publish(out_bird_msg)

    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derivatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## USED BGR AND ASSUMED IMAGE IS OF TYPE NUMPY ARRAY ##

        gs_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gs_image, (5,5), 0)
      
        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
      
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
      
        sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        binary_output = cv2.inRange(sobel_combined, thresh_min, thresh_max).astype(np.uint8)
      
        return binary_output

    def color_thresh(self, img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass

        ## USED BGR AND ASSUMED IMAGE IS OF TYPE NUMPY ARRAY ##

        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_yellow = np.array([20, 100, 70])
        upper_yellow = np.array([35, 255, 255])
        binary_output = cv2.inRange(hls_image, lower_yellow, upper_yellow)
      
        return binary_output

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs

        sobel_output = self.gradient_thresh(img)
        color_output = self.color_thresh(img)

        combined_binary = cv2.bitwise_or(sobel_output, color_output)

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(combined_binary.astype('bool'), min_size=50, connectivity=2)

        return (binaryImage * 255).astype(np.uint8)

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        width, height = img.shape[:2]
        src = np.float32([(70, 300), (50, height), (width-50, height), (width-70, 300)])
        dst = np.float32([(0, 0), (0, height), (width, height), (width, 0)])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)
        warped_img = np.uint8(cv2.warpPerspective(img, M, (height, width)))

        return warped_img, Minv, src

    def detection(self, img):
        """
        Process image to detect lane lines and generate annotated images
        """
        binary_img = self.combinedBinaryImage(img)

        img_birdeye, Minv, src = self.perspective_transform(binary_img)

        self.skip_frame += 1

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            if ret is not None:
                fit = ret['fit']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    fit = ret['fit']

                    fit = self.lane_line.add_fit(fit)

                    self.detected = True

            else:
                fit = self.lane_line.get_fit()
                
                ret = tune_fit(img_birdeye, fit)

                if ret is not None:
                    fit = ret['fit']

                    fit = self.lane_line.add_fit(fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, fit, Minv)


                if self.skip_frame > 5:
                    inference = self.yolo_model('sign_raw_image.png', verbose=False)[0].boxes

                    for box in inference:
                        if box.cls[0] == 1:  # Assuming class ID 1 is for the target sign
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(combine_fit_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                    self.skip_frame = 0

                # src_points = np.int32(src)
                # cv2.polylines(combine_fit_img, [src_points], isClosed=True, color=(0,0,255), thickness=2)

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('spatial_detector_node', anonymous=True)
    spatial_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

