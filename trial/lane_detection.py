import cv2
import numpy as np

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line

# from skimage import morphology



def get_pixel_color(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        # Access the current frame from the param
        frame = param
        
        # Get the BGR color of the pixel at the current mouse position
        color_bgr = frame[y, x]
        
        color_hls = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HLS)[0][0]
        
        # Extract HLS values
        h = color_hls[0]  # Hue (0 to 179 in OpenCV, 0 to 360 for full range)
        l = color_hls[1]  # Lightness (0 to 255)
        s = color_hls[2]  # Saturation (0 to 255)

        # Print the HSL values to the terminal
        print(f"Pixel at ({x}, {y}) - HLS: ({h}, {l}, {s})")



def create_yellow_mask(image):
    # Convert the BGR image to HLS (Hue, Lightness, Saturation)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    h = hls_image[:,:,0]
    l = hls_image[:,:,1]
    s = hls_image[:,:,2]

    hue_lower = 28
    hue_upper = 35
    
    lgt_lower = 100
    lgt_upper = 255

    sat_lower = 100
    sat_upper = 255


   
    lower_yellow = np.array([15, lgt_lower, sat_lower])  # Lower bound of yellow in HSL
    upper_yellow = np.array([35, lgt_upper, sat_upper])  # Upper bound of yellow in HSL

    # Create a mask based on the defined color range
    yellow_mask = cv2.inRange(hls_image, lower_yellow, upper_yellow)

    return yellow_mask



def perspective_transform(img, verbose=False):

    height, width = img.shape[:2]
   
    src = np.float32([(70, 300), (50, height), (width-50, height), (width-70, 300)])
    dst = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)

    warped_img = np.uint8(cv2.warpPerspective(img, M, (width, height)))

    return warped_img, M, Minv, src, dst


def detection(img):

    binary_img = create_yellow_mask(img)
    img_birdeye, M, Minv, src, dst = perspective_transform(binary_img)

    cv2.imshow('birds_eye', img_birdeye)


    detected = False
    hist = True

    lane_line = Line(n=5)

    if not hist:
        # Fit lane without previous result
        ret = line_fit(img_birdeye)
        if ret is not None:
            fit = ret['fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            lane_inds = ret['lane_inds']

    else:
        # Fit lane with previous result
        if not detected:
            ret = line_fit(img_birdeye)

            if ret is not None:
                fit = ret['fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                lane_inds = ret['lane_inds']

                fit = lane_line.add_fit(fit)

                detected = True

        else:
            fit = lane_line.get_fit()
            
            ret = tune_fit(img_birdeye, fit)

            if ret is not None:
                fit = ret['fit']
                
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']

                lane_inds = ret['lane_inds']

                fit = lane_line.add_fit(fit)

            else:
                detected = False

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


def main():

    path = '../thunder_final_project/subset.mp4'
    
    cap = cv2.VideoCapture(path)


    if not cap.isOpened():
        print("Video not opened")
        return

    paused = False
    while cap.isOpened():

        

        if not paused:
            ret, frame = cap.read()
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            if not ret:
                break  # Exit if we reach the end of the video


        # frame = cv2.GaussianBlur(frame, (5, 5), 0)


        mask = create_yellow_mask(frame)

        cv2.imshow('Frame', frame)
        

        cv2.setMouseCallback('Frame', get_pixel_color, param=frame)
        # cv2.setMouseCallback('Frame', get_pixel_color, param=mask)

        combine_fit_img, bird_fit_img = detection(frame)

        cv2.imshow('output', combine_fit_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key pressed
            # Toggle pause/resume
            paused = not paused
            if paused:
                print("Video paused. Press space to resume.")
            else:
                print("Video resumed. Press space to pause.")

        # Check if the 'q' key is pressed to quit
        if key == ord('q'):
            break

        if key == ord(']'):
            cv2.imwrite('Original.png', frame) 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

