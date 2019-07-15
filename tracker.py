# Code adapted from: 
# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv/

import cv2
import numpy as np
import dlib
from math import hypot

# Constants
BLINK_RATIO = 5.1  # Adjust if necessary to find suitable threshold (ugh why are my eyes so small)

# Initialise face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load text font
font = cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1, p2):
    '''
    Find the midpoint between two coordinates.
    '''
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blink_ratio(frame, eye_points, facial_landmarks):
    '''
    Find the ratio between the horizontal and vertical length of a specified eye.

    Args:
        frame: Frame to draw lines on eyes.
        eye_points: Array of 6 integers specifying the landmark points of an eye.
        facial_landmarks: Facial landmarks.
    
    Returns:
        Float indicating the length ratio.
    '''
    # Find left, right, top centre, and bottom centre coordinates of the specified eye
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    centre_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    centre_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Draw horizontal and vertical lines over the specified eye
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, centre_top, centre_bottom, (0, 255, 0), 2)

    # Find horizontal and vertical line length
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((centre_top[0] - centre_bottom[0]), (centre_top[1] - centre_bottom[1]))

    # Return ratio between horizontal and vertical line length
    ratio = hor_line_length / ver_line_length
    return ratio

def main():
    cap = cv2.VideoCapture(0)  # Camera capture

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grey)
        for face in faces:
            # Find face landmarks
            landmarks = predictor(grey, face)

            # Find blink ratio of each eye
            left_eye_ratio = get_blink_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)  # Left eye consists of points 36, 37, 38, 39, 40, 41
            right_eye_ratio = get_blink_ratio(frame, [42, 43, 44, 45, 46, 47], landmarks)  # Right eye consists of points 42, 43, 44, 45, 46, 47

            # Find total blink ratio as an average of both eyes
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # Detect blink
            if blink_ratio > BLINK_RATIO:
                cv2.putText(frame, "Blink detected", (50, 150), font, 3, (255, 0, 0))

        cv2.imshow("Webcam capture", frame)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()