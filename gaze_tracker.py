import os
from imutils import face_utils
import cv2
import dlib
from eye import Eye

class GazeTracker(object):
    # Constants
    # Number of consecutive frames the eye must be below threshold
    EAR_CONSEC_FRAMES = 3

    def __init__(self):
        self.frame = None

        # Initialise face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self.predictor = dlib.shape_predictor(model_path)

        # Indexes for facial landmarks of left and right eyes
        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Initialise frame counter and total number of blinks
        self.ear_frame_count = 0
        self.total_blinks = 0
    
    def refresh(self, frame):
        '''
        Refreshes the frame.

        Args:
            frame (numpy.ndarray): The frame to analyse.
        '''
        self.frame = frame
    
    def track(self, ear_thresh, binar_thresh):
        '''
        Tracks user gaze and shows information on the frame.

        Args:
            ear_thresh (float): Eye aspect ratio threshold value.
            binar_thresh (int): Binarise threshold value.
        '''

        grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(grey)
        for face in faces:
            # Find face landmarks and convert coordinates to a NumPy array
            landmarks = self.predictor(grey, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Extract left and right eye coordinates
            left_eye_coords = landmarks[self.l_start:self.l_end]
            right_eye_coords = landmarks[self.r_start:self.r_end]

            # Initialise left and right eye objects
            left_eye = Eye(self.frame, left_eye_coords)
            right_eye = Eye(self.frame, right_eye_coords)
            
            # Compute eye aspect ratio for both eyes
            left_ear = left_eye.eye_aspect_ratio()
            right_ear = right_eye.eye_aspect_ratio()

            # Find total eye aspect ratio as an average of both eyes
            ear = (left_ear + right_ear) / 2.0

            # Draw detected eye outlines onto face frame
            left_eye.draw()
            right_eye.draw()

            # Isolate eyes and show binary images in separate windows
            left_eye_frame = left_eye.isolate(binar_thresh)
            right_eye_frame = right_eye.isolate(binar_thresh)
            cv2.imshow("left eye", left_eye_frame)
            cv2.imshow("right eye", right_eye_frame)

            # Compute gaze ratio for both eyes
            left_gr = left_eye.gaze_ratio(left_eye_frame)
            right_gr = right_eye.gaze_ratio(right_eye_frame)

            # Find total gaze ratio as an average of both eyes
            gr = (left_gr + right_gr) / 2.0

            # Increment blink frame counter if eye aspect ratio is below threshold
            if ear < ear_thresh:
                self.ear_frame_count += 1
            else:
                # Increment total number of blinks if eyes closed for a sufficient number of frames
                if self.ear_frame_count >= self.EAR_CONSEC_FRAMES:
                    self.total_blinks += 1
                    cv2.putText(self.frame, "BLINK", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                # Gaze detection
                elif gr <= 0.7:  # Need to adjust threshold
                    cv2.putText(self.frame, "RIGHT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                elif 0.7 < gr < 1.5:  # Need to adjust threshold
                    cv2.putText(self.frame, "CENTER", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    cv2.putText(self.frame, "LEFT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                self.ear_frame_count = 0  # Reset blink frame counter
            
            # Display total number of blinks, eye aspect ratio, and gaze ratio
            cv2.putText(self.frame, f"Blinks: {self.total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.frame, f"Gaze ratio: {gr:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam capture", self.frame)