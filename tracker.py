# Adapted from https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

import cv2
import numpy as np

# Initialise haar cascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Initialise blob detection algorithm
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500  # Area filtering for better results
detector = cv2.SimpleBlobDetector_create(detector_params)

# Detect face
def detect_face(img, cascade):
    # Make image frame grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(
        img_grey,
        scaleFactor = 1.1,  # Need to experiment with this value depending on camera
        minNeighbors = 5
    )

    # Only return largest face frame to mitigate false detections
    if len(faces) > 1:
        largest = (0, 0, 0, 0)
        for i in faces:
            if i[3] > largest[3]:
                largest = i
        largest = np.array([i], np.int32)
    elif len(faces) == 1:
        largest = faces
    else:
        return None
    for (x, y, w, h) in largest:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 2)  # TEMPORARY CODE draw a rectangle around face
        frame = img[y: y + h, x: x + w]
    
    return frame

# Detect eyes
def detect_eyes(img, cascade):
    # Make image frame grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = cascade.detectMultiScale(
        img_grey,
        scaleFactor = 1.3,  # Need to experiment with this value depending on camera
        minNeighbors = 5
    )

    # Get face frame width and height
    width = np.size(img, 1)
    height = np.size(img, 0)

    # Pre-define left and right eye variables in case they are not found
    left_eye = None
    right_eye = None

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # TEMPORARY CODE draw a rectangle around eyes
        if y > height / 2:  # pass if eye is at bottom
            pass
        eye_centre = x + w / 2
        if eye_centre < width * 0.5:
            left_eye = img[y: y + h, x: x + w]
        else:
            right_eye = img[y: y + h, x: x + w]
    
    return left_eye, right_eye

# Cut our eyebrows from eye frame
def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)  # Eyebrows take up ~25% of image starting from top
    img = img[eyebrow_h: height, 0: width]  # Cut eyebrows out

    return img

# Detect and draw blobs on frames
def blob_process(img, threshold, detector):
    frame_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(
        frame_grey,
        thresh = threshold,
        maxval = 255,
        type = cv2.THRESH_BINARY
    )  # _ is a throwaway variable
    
    # Reduce 'noise'
    img = cv2.erode(img, None, iterations = 2)
    img = cv2.dilate(img, None, iterations = 4)
    img = cv2.medianBlur(img, 5)

    keypoints = detector.detect(img)
    return keypoints

# Trackbar requires a function to happen on every movement
# Create a function that does nothing
def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)  # Camera capture
    cv2.namedWindow("Webcam capture")
    cv2.createTrackbar("Threshold", "Webcam capture", 0, 255, nothing)

    while True:
        _, frame = cap.read()  # _ is a throwaway variable
        face_frame = detect_face(frame, face_cascade)
        if face_frame is not None:  # Prevent crash if face not detected
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:  # Prevent crash if eye not detected
                    threshold = cv2.getTrackbarPos("Threshold", "Webcam capture")
                    eye = cut_eyebrows(eye)  # Cut out eyebrows from eye frame
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("Webcam capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()