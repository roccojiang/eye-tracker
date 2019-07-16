# Code adapted from: 
# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv/
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

from scipy.spatial import distance
from math import hypot
from imutils import face_utils
import cv2
import imutils
import numpy as np
import dlib

# Constants
# Default eye aspect ratio to indicate a blink - adjust if necessary
EYE_AR_THRESH = 0.25  # For Rocco: ~0.25 for intentional blinks (ugh why are my eyes so small)
# Number of consecutive frames the eye must be below threshold
EYE_AR_CONSEC_FRAMES = 3

# Initialise face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes for facial landmarks of left and right eyes
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Load text font
font = cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1, p2):
    '''
    Find the midpoint between two coordinates.

    Args:
        p1: First coordinate point as a NumPy array.
        p2: Second coordinate point as a NumPy array.
    
    Returns:
        The midpoint between the two coordinates as a tuple.
    '''
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)  # (x, y)

def eye_aspect_ratio(eye):
    '''
    Computes the eye aspect ratio as defined in Soukupová and Čech, 2016

    Args:
        eye: Array of 6 integers specifying the facial landmark coordinates for a given eye.
    
    Returns:
        The eye aspect ratio as a float.
    '''
    # Compute euclidean distances between two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute euclidean distance between horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Compute and return eye aspect ratio
    ear = (A + B) / (C * 2.0)
    return ear

def gaze_ratio(eye):
    '''
    Computes the gaze ratio between the white pixels in the left and right half of the eye.

    Args:
        eye: Array of 6 integers specifying the facial landmark coordinates for a given eye.
    
    Returns:
        The gaze ratio as a float.
    '''
    eye_region = np.array([(eye[0][0], eye[0][1]),
                           (eye[1][0], eye[1][1]),
                           (eye[2][0], eye[2][1]),
                           (eye[3][0], eye[3][1]),
                           (eye[4][0], eye[4][1]),
                           (eye[5][0], eye[5][1])], dtype=np.int32)

    # Create mask of inside of eye and exclude surroundings
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye_mask = cv2.bitwise_and(grey, grey, mask=mask)

    # Cut out rectangular shape using the extreme points of the eye
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # Process image
    grey_eye = eye_mask[min_y: max_y, min_x: max_x]
    _, processed_eye = cv2.threshold(grey_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = processed_eye.shape

    # Count white pixels on left and right side of eye
    left_side_threshold = processed_eye[0:height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = processed_eye[0:height, int(width/2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Calculate gaze ratio
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio

def draw_eye(frame, eye):
    '''
    Draw detected eyes onto the frame.

    Args:
        frame: Frame to draw onto.
        eye: Array of 6 integers specifying the facial landmark coordinates for a given eye.
    '''
    # Compute eye hull and draw contours
    eye_hull = cv2.convexHull(eye)
    cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)

    # Express left, right, top centre, and bottom centre coordinates as tuples
    left_point = (eye[0][0], eye[0][1])
    right_point = (eye[3][0], eye[3][1])
    centre_top = midpoint(eye[1], eye[2])
    centre_bottom = midpoint(eye[5], eye[4])

    # Draw horizontal and vertical lines over eye
    cv2.line(frame, left_point, right_point, (255, 255, 0), 1)  # Horizontal line through left and right points
    cv2.line(frame, centre_top, centre_bottom, (255, 255, 0), 1)  # Vertical line through top and bottom midpoints

# Trackbar requires a function to happen on every movement
# So this function does nothing
def nothing(x):
    pass

def main():
    # Initialise frame counter and total number of blinks
    COUNT = 0
    TOTAL = 0

    cap = cv2.VideoCapture(0)  # Camera capture
    cv2.namedWindow("Webcam capture")
    cv2.createTrackbar("Threshold", "Webcam capture", 25, 40, nothing)

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        frame = imutils.resize(frame, width=720)  # Downsize window to significantly reduce lag
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar to change eye aspect ratio threshold value
        # OpenCV only allows integer values so everything is multiplied by 100
        EYE_AR_THRESH = cv2.getTrackbarPos("Threshold", "Webcam capture") / 100.0

        faces = detector(grey)
        for face in faces:
            # Find face landmarks and convert coordinates to a NumPy array
            landmarks = predictor(grey, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Extract left and right eye coordinates
            left_eye = landmarks[l_start:l_end]
            right_eye = landmarks[r_start:r_end]
            
            # Compute eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Find total eye aspect ratio as an average of both eyes
            ear = (left_ear + right_ear) / 2.0

            # Draw eyes
            draw_eye(frame, left_eye)
            draw_eye(frame, right_eye)

            # Compute gaze ratio for both eyes
            left_gr = gaze_ratio(left_eye)
            right_gr = gaze_ratio(right_eye)

            # Find total gaze ratio as an average of both eyes
            gr = (left_gr + right_gr) / 2.0

            # Increment blink frame counter if eye aspect ratio is below threshold
            if ear < EYE_AR_THRESH:
                COUNT += 1
            else:
                # Increment total number of blinks if eyes closed for a sufficient number of frames
                if COUNT >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNT = 0  # Reset blink frame counter
            
            # Display total number of blinks, eye aspect ratio, and gaze ratio
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Gaze ratio: {gr:.2f}", (300, 60), font, 1, (0, 0, 255), 2)

            # Gaze detection
            if gr <= 0.5:  # Need to ajust threshold
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            elif 0.5 < gr < 1.7:  # Need to adjust threshold
                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

        cv2.imshow("Webcam capture", frame)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()