import cv2
import numpy as np

class WaypointGenerator:

    def __init__(self):
        
        self.waypoints = []
        
        self.y = np.array([330, 360, 390, 420, 450])


    def compute_waypoints(self, coeff):

        self.waypoints = []
        for y in self.y:
            x = coeff[2] + coeff[1] * y + coeff[0] * y**2
            waypoint = (int(x), y)
            self.waypoints.append(waypoint)

        return self.waypoints
    

    def waypoint_viz(self, undist, fit, m_inv):

        # # Generate x and y values for plotting
        # ploty = self.y
        # fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]


        # Create an image to draw the lines on
        color_warp = np.zeros_like(undist).astype(np.uint8)


        # Recast the x and y points into usable format for cv2.fillPoly()
        # pts = np.array([np.transpose(np.vstack([fitx, ploty]))])


        # Draw the lane onto the warped blank image
        for waypoint in self.waypoints:
            cv2.circle(color_warp, waypoint, 5, (65,103,274), -1)
        # print("finished drawing waypoints")
        # cv2.polylines(color_warp, np.int32([pts]), isClosed=False, color=(0, 255, 0), thickness=15)


        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))


        # Combine the result with the original image
        # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


        return newwarp