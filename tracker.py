# Code adapted from: 
# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv/

import cv2
import numpy as np
import dlib

# Initialise face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    '''
    Find the midpoint between two coordinates.
    '''
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def main():
    cap = cv2.VideoCapture(0)  # Camera capture

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grey)
        for face in faces:
            # Find face landmarks
            landmarks = predictor(grey, face)

            # Left eye
            leye_left_point = (landmarks.part(36).x, landmarks.part(36).y)  # Left point of left eye at point 36
            leye_right_point = (landmarks.part(39).x, landmarks.part(39).y)  # Right point of left eye at point 39
            leye_centre_top = midpoint(landmarks.part(37), landmarks.part(38))  # Midpoint of top of left eye between points 37 and 38
            leye_centre_bottom = midpoint(landmarks.part(41), landmarks.part(40))  # Midpoint of bottom of left eye between points 41 and 40

            leye_hor_line = cv2.line(frame, leye_left_point, leye_right_point, (0, 255, 0), 2)
            leye_ver_line = cv2.line(frame, leye_centre_top, leye_centre_bottom, (0, 255, 0), 2)

            # Right eye
            reye_left_point = (landmarks.part(42).x, landmarks.part(42).y)  # Left point of right eye at point 42
            reye_right_point = (landmarks.part(45).x, landmarks.part(45).y)  # Right point of right eye at point 45
            reye_centre_top = midpoint(landmarks.part(43), landmarks.part(44))  # Midpoint of top of right eye between points 43 and 44
            reye_centre_bottom = midpoint(landmarks.part(47), landmarks.part(46))  # Midpoint of bottom of right eye between points 47 and 46

            reye_hor_line = cv2.line(frame, reye_left_point, reye_right_point, (255, 255, 0), 2)
            reye_ver_line = cv2.line(frame, reye_centre_top, reye_centre_bottom, (255, 255, 0), 2)
        
        cv2.imshow("Webcam capture", frame)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()