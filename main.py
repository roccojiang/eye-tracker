# Code adapted from: 
# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv/
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# https://github.com/antoinelame/GazeTracking

import cv2
import imutils
from eye import Eye
from gaze_tracker import GazeTracker

# Trackbar requires a function to happen on every movement
# So this function does nothing
def nothing(x):
    pass

def main():
    # Default theshold values
    EAR_THRESH = 0.25
    BINAR_THRESH = 45

    cap = cv2.VideoCapture(0)  # Camera capture
    cv2.namedWindow("Webcam capture")
    cv2.createTrackbar("EAR threshold", "Webcam capture", 25, 40, nothing)
    cv2.createTrackbar("Binarise threshold", "Webcam capture", 45, 100, nothing)

    gaze = GazeTracker()

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        frame = imutils.resize(frame, width=720)  # Downsize window to significantly reduce lag

        # Trackbar to change eye aspect ratio threshold value
        # OpenCV only allows integer values so everything is multiplied by 100
        EAR_THRESH = cv2.getTrackbarPos("EAR threshold", "Webcam capture") / 100.0
        # Trackbar to change binarise threshold value
        BINAR_THRESH = cv2.getTrackbarPos("Binarise threshold", "Webcam capture")

        gaze.refresh(frame)
        gaze.track(EAR_THRESH, BINAR_THRESH)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()