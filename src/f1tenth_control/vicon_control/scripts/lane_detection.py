import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.sub_image = rospy.Subscriber('D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.lane_line = Line(n=5)

        self.detected = False
        self.hist = True

        self.image_saved = False


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if not self.image_saved:
            cv2.imwrite('test.png', cv_image)
            self.image_saved = True

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        cv2.imwrite('raw_image.png', raw_img)
        cv2.imwrite('bird.png', bird_image)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO

        ## USED BGR AND ASSUMED IMAGE IS OF TYPE NUMPY ARRAY ##

        gs_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gs_image, (5,5), 0)
        
        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        
        sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

        binary_output = cv2.inRange(sobel_combined, thresh_min, thresh_max).astype(np.uint8)

        ####

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO

        ## USED BGR AND ASSUMED IMAGE IS OF TYPE NUMPY ARRAY ##

        # blurred = cv2.GaussianBlur(img, (3,3), 0)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        binary_output = np.zeros_like(hsv_img[:, :, 0], dtype=np.uint8)
        binary_output[(hsv_img[:, :, 1] >= 100) & (hsv_img[:, :, 1] <= 250) &
                        (hsv_img[:, :, 2] >= 100) & (hsv_img[:, :, 0] <= 35)] = 1

        ####

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        sobel_output = self.gradient_thresh(img)
        color_output = self.color_thresh(img)

        ####

        combined_binary = np.zeros_like(sobel_output)
        combined_binary[(sobel_output == 1) | (color_output == 1)] = 1

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(combined_binary.astype('bool'),min_size=50,connectivity=2)

        binaryImage = (binaryImage * 255).astype(np.uint8)

        # cv2.imshow("hello", binaryImage)

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO

        ####
        
                # pts1 = np.float32([[58, 420], [560, 420],
        #                [225, 309], [403, 309]])
        # pts2 = np.float32([[58, 420], [560, 420],
        #                 [58, 309], [560, 309]])
        
        # pts2 = np.float32([[58, 309], [560, 309],
        #                 [58, 420], [560, 420]])

        width, height = img.shape[:2]
        
        # print(width, height)

        # for gazebo
        # pts1 = np.float32([[299, 250], [340, 250], [30, 400], [600, 400]])
        
        # pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

        # pts1 = np.float32([[284, 260], [355, 260], [30, 400], [600, 400]])
        
        # pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        
        # rosbag
        # pts1 = np.float32([[280, 290], [354, 290], [167, 458], [418, 458]])
        
        # pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

        # pts1 = np.float32([[100, 300], [100, 900], [375, 300], [375, 900]])

        # if height > 700:
        #     pts1 = np.float32([[556, 360], [670, 360], [100, 700], [1100, 700]])
        # else: 
        #     pts1 = np.float32([[540, 210], [700, 210], [200, 360], [940, 360]])
            
        # pts2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

        src = np.float32([(110, 250), (530, 250), (110, 440), (530, 440)])

        

        dst = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        
        # ENTER HERE

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = np.uint8(cv2.warpPerspective(img, M, (height, width)))

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
                combine_fit_img = final_viz(img, fit, Minv)


                src_points = np.int32(src)
                cv2.polylines(combine_fit_img, [src_points], isClosed=True, color=(0,0,255), thickness=2)

            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

    # detector = lanenet_detector()
    
    # # Path to the image file
    # image_path = 'test.png'
    
    # # Load the image
    # image = cv2.imread(image_path)
    
    # if image is None:
    #     print(f"Error: Unable to load image at {image_path}")
    # else:
    #     # Show the original image
    #     cv2.imshow('Original Image', image)

    #     # Run the gradient_thresh method using the class object 'detector'
    #     combined_image = detector.combinedBinaryImage(image)

    #     output_image = detector.perspective_transform(combined_image)

    #     output2 = detector.perspective_transform(image)

    #     # Show the processed image
    #     cv2.imshow('Processed Image', output_image[0])
    #     cv2.imshow('Processed Image 2', output2[0])

    #     # Wait until a key is pressed to close the image windows
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
