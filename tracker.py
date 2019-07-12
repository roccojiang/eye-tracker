# Adapted from https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

import cv2
import numpy as np

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

def main():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    img = cv2.imread("test_image.jpg")

    detect_face(img, face_cascade)
    detect_eyes(img, eye_cascade)

    cv2.imshow("my image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()