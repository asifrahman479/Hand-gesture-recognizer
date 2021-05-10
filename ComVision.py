import constants
import cv2
import numpy as np
import copy
import math
from collections import deque

# needed for motion tracking
pts = deque(maxlen=constants.TRACKING_BUFFER_MAX_LEN)
frame_counter = 0
(diffX, diffY) = (0, 0)
direction = "None"

def make_frame_smaller(frame,ratio):
    # get size from a matrix
    height = frame.shape[0]
    width = frame.shape[1]
    #resize using cv2.resize(...)
    result = cv2.resize(frame,(int(width*ratio),int(height*ratio)))
    return result

def background_removal(frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    rmVer = cv2.bitwise_and(frame, frame, mask=fgmask)
    return rmVer

def skin_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)
    _, thresh = cv2.threshold(
        blur,
        constants.THRESHOLD_MIN,
        constants.THRESHOLD_MAX,
        cv2.THRESH_BINARY
    )
    cv2.imshow("thresh",thresh)
    return thresh


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

def detect_waving(cnt, drawing):
    global direction

    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
    M = cv2.moments(cnt)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if radius > 10:
        cv2.circle(drawing, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
    direction = calculate_direction(drawing)
    if direction == "East" or "West":
        return True
    return False

def calculate_direction(drawing):
    global direction

	# loop over the tracked centers
    for i in np.arange(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        if frame_counter >= 10 and i == 1 and pts[-10] is not None:
            diffX = pts[-10][0] - pts[i][0]
            diffY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ("", "")

            if np.abs(diffX) > 20:
                dirX = "East" if np.sign(diffX) == 1 else "West"

            if np.abs(diffY) > 20:
                dirY = "North" if np.sign(diffY) == 1 else "South"

            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY
        
        thickness = int(np.sqrt(constants.TRACKING_BUFFER_MAX_LEN / float(i + 1)) * 2.5)
        cv2.line(drawing, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    cv2.putText(drawing, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 255), 3)
    return direction

#You don't have to modify the code here, it simply reads the input from your camera, and run functions above to give an
#image showing where your object is.

camera = cv2.VideoCapture(0)
camera.set(10, 200)
BGet = False
gestureStart = False

while camera.isOpened():
    ret, frame = camera.read()

    #make your frames smaller
    frame = make_frame_smaller(frame, 0.7)
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    cv2.imshow('original', frame)

    #remove your back ground
    if BGet:
        bgRM = background_removal(frame)
        thresh = skin_detection(bgRM)
        cv2.imshow('mask', bgRM)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) != 0):  # draws the contours of that object which has the highest
            res = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(res)
            drawing = np.zeros(bgRM.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal, cnt = calculateFingers(res, drawing)
            if gestureStart:
                if isFinishCal and cnt <= 5:
                    if cnt == 1:
                        cv2.putText(frame, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                    elif cnt == 2:
                        cv2.putText(frame, "Trident Strike!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    elif cnt == 3:
                        cv2.putText(frame, "Where is your thumb?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    elif cnt >= 4:
                        waving = detect_waving(res, drawing)
                        frame_counter += 1
                        if not waving:
                            cv2.putText(frame, "Wanna Try Moving?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        else:
                            cv2.putText(frame, "Your Waving!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "Fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('output',np.hstack([frame,drawing]))

    #press q to stop capturing frames
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k & 0xFF == ord('b'):  # get a base background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        BGet = True
        frame_counter = 0
        gestureStart = True
    elif k & 0xFF == ord('s'):  # pause
        bgModel = False
        gestureStart = False
        BGet = False
