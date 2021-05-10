import numpy as np
import cv2
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
LOWER = np.array([0,20, 70], dtype = "uint8")
UPPER = np.array([20, 255, 255], dtype = "uint8")

THRESHOLD_MIN = 60
THRESHOLD_MAX = 255

TRACKING_BUFFER_MAX_LEN = 30
