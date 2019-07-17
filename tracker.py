# Code adapted from: 
# https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv/
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# https://github.com/rj14ng/eye-tracking-car

from imutils import face_utils
import cv2
import imutils
import dlib
from eye import Eye

# Constants
# Number of consecutive frames the eye must be below threshold
EAR_CONSEC_FRAMES = 3

# Initialise face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes for facial landmarks of left and right eyes
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Load text font
font = cv2.FONT_HERSHEY_SIMPLEX

# Trackbar requires a function to happen on every movement
# So this function does nothing
def nothing(x):
    pass

def main():
    # Initialise frame counter and total number of blinks
    COUNT = 0
    TOTAL = 0

    # Default theshold values
    EAR_THRESH = 0.25
    BINAR_THRESH = 45

    cap = cv2.VideoCapture(0)  # Camera capture
    cv2.namedWindow("Webcam capture")
    cv2.createTrackbar("EAR threshold", "Webcam capture", 25, 40, nothing)
    cv2.createTrackbar("Binarise threshold", "Webcam capture", 45, 100, nothing)

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        frame = imutils.resize(frame, width=720)  # Downsize window to significantly reduce lag
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar to change eye aspect ratio threshold value
        # OpenCV only allows integer values so everything is multiplied by 100
        EAR_THRESH = cv2.getTrackbarPos("EAR threshold", "Webcam capture") / 100.0
        # Trackbar to change binarise threshold value
        BINAR_THRESH = cv2.getTrackbarPos("Binarise threshold", "Webcam capture")

        faces = detector(grey)
        for face in faces:
            # Find face landmarks and convert coordinates to a NumPy array
            landmarks = predictor(grey, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Extract left and right eye coordinates
            left_eye_coords = landmarks[l_start:l_end]
            right_eye_coords = landmarks[r_start:r_end]

            # Initialise left and right eye objects
            left_eye = Eye(frame, left_eye_coords)
            right_eye = Eye(frame, right_eye_coords)
            
            # Compute eye aspect ratio for both eyes
            left_ear = left_eye.eye_aspect_ratio()
            right_ear = right_eye.eye_aspect_ratio()

            # Find total eye aspect ratio as an average of both eyes
            ear = (left_ear + right_ear) / 2.0

            # Draw detected eye outlines onto face frame
            left_eye.draw()
            right_eye.draw()

            # Isolate eyes and show binary images in separate windows
            left_eye_frame = left_eye.isolate(BINAR_THRESH)
            right_eye_frame = right_eye.isolate(BINAR_THRESH)
            cv2.imshow("left eye", left_eye_frame)
            cv2.imshow("right eye", right_eye_frame)

            # Compute gaze ratio for both eyes
            left_gr = left_eye.gaze_ratio(left_eye_frame)
            right_gr = right_eye.gaze_ratio(right_eye_frame)

            # Find total gaze ratio as an average of both eyes
            gr = (left_gr + right_gr) / 2.0

            # Increment blink frame counter if eye aspect ratio is below threshold
            if ear < EAR_THRESH:
                COUNT += 1
            else:
                # Increment total number of blinks if eyes closed for a sufficient number of frames
                if COUNT >= EAR_CONSEC_FRAMES:
                    TOTAL += 1
                    cv2.putText(frame, "BLINK", (30, 100), font, 2, (0, 0, 255), 3)
                
                # Gaze detection
                elif gr <= 0.7:  # Need to ajust threshold
                    cv2.putText(frame, "RIGHT", (30, 100), font, 2, (0, 0, 255), 3)
                elif 0.7 < gr < 1.5:  # Need to adjust threshold
                    cv2.putText(frame, "CENTER", (30, 100), font, 2, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "LEFT", (30, 100), font, 2, (0, 0, 255), 3)

                COUNT = 0  # Reset blink frame counter
            
            # Display total number of blinks, eye aspect ratio, and gaze ratio
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Gaze ratio: {gr:.2f}", (300, 60), font, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam capture", frame)

        if cv2.waitKey(1) == 27:  # Esc
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()