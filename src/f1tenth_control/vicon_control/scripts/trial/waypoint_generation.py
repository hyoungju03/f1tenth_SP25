import cv2
import numpy as np


def filter(image):

    # Check if the image was loaded properly
    if image is None:
        print("Error: Unable to load image.")
        exit()

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create a binary mask where yellow colors are white and the rest are black
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Optional: Remove small blobs (noise) using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Save the binary image
    return mask_cleaned


def generate_waypoints(original, binary_image):

    waypoints = list()

    if binary_image is None:
        print("Error: Unable to load binary image.")
        exit()

    # Load the original image (to draw waypoints)
    original_image = original
    if original_image is None:
        print("Error: Unable to load original image.")
        exit()

    width, height = binary_image.shape[:2]

    y_coords = [3*width//4+(i-1)*30 for i in range(5)]

    for y in y_coords:
        row = binary_image[y,:]
        first = 0
        count = 0
        for i in range(len(row)):
            if row[i] == 255:
                if count == 0:
                    first = i
                count += 1
            if row[i] == 0 and count != 0:
                break
        x = first + count//2
        point = (x, y)
        waypoints.append(point)
        cv2.circle(original_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # # Save the output image
    cv2.imwrite('./output.png', original_image)
    waypoints.reverse()

    print(waypoints)

    return waypoints

def scale_to_global(waypoints):
    
    for waypoint in waypoints:
        print(waypoint)



def main():
    path = './test.png'
    image = cv2.imread(path)
    window_name = "test image"
    windw1 = "binary"

    masked = filter(image)

    cv2.imshow(window_name, image)
    cv2.imshow(windw1, masked)

    waypoints = generate_waypoints(image, masked)
    # scale_to_global(waypoints)



    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()