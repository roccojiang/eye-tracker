# Eye Tracker

A Python program that uses a webcam to track your gaze, detecting blinks and gaze direction (left or right).

## Requirements
- Python 3.7+
- dlib 19.17+
- OpenCV 3.4
- imutils (https://github.com/jrosebr1/imutils)
- NumPy 1.15.2+
- SciPy 0.14+

Dependencies can be installed using
```
pip install -r requirements.txt
```

## Guide
To run:
```
python3 main.py
```

There are two sliders where you can change threshold values to match your eye size and lighting conditions:
- `EAR threshold` (eye aspect ratio) changes the threshold value to detect blinking
- `Binarise threshold` changes the threshold to detect pupils, and should be changed depending on lighting conditions; the binarised images of the left and right eye will be shown in separate windows so the optimal value can be found by the user