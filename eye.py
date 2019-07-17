from scipy.spatial import distance
from math import hypot
import cv2
import numpy as np

class Eye(object):
    def __init__(self, frame, coordinates):
        self.frame = frame
        self.grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.coordinates = coordinates
    
    @staticmethod
    def midpoint(p1, p2):
        '''
        Find the midpoint between two coordinates.

        Args:
            p1 (numpy.ndarray): First coordinate point.
            p2 (numpy.ndarray): Second coordinate point.
        
        Returns:
            (tuple): Midpoint between the two coordinates.
        '''
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def eye_aspect_ratio(self):
        '''
        Computes the eye aspect ratio as defined in Soukupová and Čech (2016), in order to detect blinks.
        
        Returns:
            ear (float): Eye aspect ratio.
        '''
        # Compute euclidean distances between two sets of vertical eye landmarks
        A = distance.euclidean(self.coordinates[1], self.coordinates[5])
        B = distance.euclidean(self.coordinates[2], self.coordinates[4])

        # Compute euclidean distance between horizontal eye landmark
        C = distance.euclidean(self.coordinates[0], self.coordinates[3])

        # Compute and return eye aspect ratio
        ear = (A + B) / (C * 2.0)
        return ear
    
    def isolate(self, threshold):
        '''
        Isolates an eye from a face frame.

        Args:
            threshold (int): Threshold value to create binary image.
        
        Returns:
            threshold_eye (numpy.ndarray): Binary image of the eye.
        '''
        eye_region = np.array([(self.coordinates[0][0], self.coordinates[0][1]),
                               (self.coordinates[1][0], self.coordinates[1][1]),
                               (self.coordinates[2][0], self.coordinates[2][1]),
                               (self.coordinates[3][0], self.coordinates[3][1]),
                               (self.coordinates[4][0], self.coordinates[4][1]),
                               (self.coordinates[5][0], self.coordinates[5][1])], dtype=np.int32)

        # Create mask of inside of eye and exclude surroundings
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye_mask = cv2.bitwise_and(self.grey, self.grey, mask=mask)

        # Cut out rectangular shape using the extreme points of the eye
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        # Process image
        grey_eye = eye_mask[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(grey_eye, threshold, 255, cv2.THRESH_BINARY)
        
        return threshold_eye
    
    def gaze_ratio(self, threshold_eye):
        '''
        Computes the gaze ratio between the white pixels in the left and right half of the eye.

        Args:
            threshold_eye (numpy.ndarray): Binary image eye frame returned from the isolate() method.
        
        Returns:
            gaze_ratio (float): Gaze ratio.
        '''

        # Find height and width
        height, width = threshold_eye.shape

        # Count white pixels on left and right side of eye
        left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0:height, int(width/2):width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        # Calculate gaze ratio
        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white

        return gaze_ratio
    
    def draw(self):
        '''
        Draw detected eyes onto the frame.
        '''
        # Compute eye hull and draw contours
        eye_hull = cv2.convexHull(self.coordinates)
        cv2.drawContours(self.frame, [eye_hull], -1, (0, 255, 0), 1)

        # Express left, right, top centre, and bottom centre coordinates as tuples
        left_point = (self.coordinates[0][0], self.coordinates[0][1])
        right_point = (self.coordinates[3][0], self.coordinates[3][1])
        centre_top = self.midpoint(self.coordinates[1], self.coordinates[2])
        centre_bottom = self.midpoint(self.coordinates[5], self.coordinates[4])

        # Draw horizontal and vertical lines over eye
        cv2.line(self.frame, left_point, right_point, (255, 255, 0), 1)  # Horizontal line through left and right points
        cv2.line(self.frame, centre_top, centre_bottom, (255, 255, 0), 1)  # Vertical line through top and bottom midpoints