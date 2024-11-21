import numpy as np
import cv2
import matplotlib.pyplot as plt


def line_fit(binary_warped):
    """
    Find and fit a single lane line
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    # Find the peak of the histogram
    x_base = np.argmax(histogram)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current position to be updated for each window
    x_current = x_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty list to receive lane pixel indices
    lane_inds = []


    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y
        y_low = binary_warped.shape[0] - (window + 1) * window_height
        y_high = binary_warped.shape[0] - window * window_height
        x_low = x_current - margin
        x_high = x_current + margin


        # Draw the window on the visualization image
        cv2.rectangle(out_img, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)


        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                     (nonzerox >= x_low) & (nonzerox < x_high)).nonzero()[0]


        # Append these indices to the list
        lane_inds.append(good_inds)


        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int32(np.mean(nonzerox[good_inds]))


    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)


    # Extract lane pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]


    # Fit a second order polynomial to the lane line
    try:
        fit = np.polyfit(y, x, 2)
    except TypeError:
        print("Unable to detect lane")
        return None


    # Return a dict of relevant variables
    ret = {}
    ret['fit'] = fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['lane_inds'] = lane_inds
    return ret


def tune_fit(binary_warped, fit):
    """
    Given a previously fit line, quickly try to find the line based on previous lines
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) &
                 (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))


    # Extract lane line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]


    # If we don't find enough relevant points, return None
    min_inds = 10
    if y.shape[0] < min_inds:
        return None


    # Fit a second order polynomial
    fit = np.polyfit(y, x, 2)


    # Return a dict of relevant variables
    ret = {}
    ret['fit'] = fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['lane_inds'] = lane_inds
    return ret


def bird_fit(binary_warped, ret, save_file=None):
    """
    Visualize the predicted lane line with margin, on binary warped image
    """
    # Grab variables from ret dictionary
    fit = ret['fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    lane_inds = ret['lane_inds']


    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in lane line pixels
    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]


    # Generate a polygon to illustrate the search window area
    margin = 100  # Keep in sync with *_fit()
    line_window1 = np.array([np.transpose(np.vstack([fitx - margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin, ploty])))])
    line_pts = np.hstack((line_window1, line_window2))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    return result


def final_viz(undist, fit, m_inv):
    """
    Final lane line prediction visualized and overlayed on top of original image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]


    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)


    # Recast the x and y points into usable format for cv2.fillPoly()
    pts = np.array([np.transpose(np.vstack([fitx, ploty]))])

    poly_points_birdseye = np.hstack([fitx.reshape(-1,1), ploty.reshape(-1,1)])
    poly_points_birdseye = np.array([poly_points_birdseye], dtype=np.float32)
    poly_points_normal_view = cv2.perspectiveTransform(poly_points_birdseye, m_inv)


    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int32([pts]), isClosed=False, color=(0, 255, 0), thickness=15)


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))


    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


    return result, poly_points_normal_view





