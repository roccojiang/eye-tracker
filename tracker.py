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
# Eye aspect ratio to indicate a blink - adjust if necessary
EYE_AR_THRESH = 0.23  # Rocco - 0.23 for intentional blinks (ugh why are my eyes so small)
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

def main():
    # Initialise frame counter and total number of blinks
    COUNT = 0
    TOTAL = 0

    cap = cv2.VideoCapture(0)  # Camera capture

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        frame = imutils.resize(frame, width=720)  # Downsize window to significantly reduce lag
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

            # Increment blink frame counter if eye aspect ratio is below threshold
            if ear < EYE_AR_THRESH:
                COUNT += 1
            else:
                # Increment total number of blinks if eyes closed for a sufficient number of frames
                if COUNT >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNT = 0  # Reset blink frame counter
            
            # Display total number of blinks and computed eye aspect ratio
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), font, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam capture", frame)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()