import numpy as np
import cv2
import sys
from time import time
import pandas as pd

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01

displacement = [["cx","cy","peak value"]]

def drawBox(img, bbox):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Box drawing function
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 0), 3, 1)
    cv2.putText(img, "Tracking", (15, 70), font, 0.5, (0, 0, 255), 2)

# mouse callback function

def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True



if __name__ == '__main__':

    if (len(sys.argv) == 1):
        cap = cv2.VideoCapture(0)
    elif (len(sys.argv) == 2):
        if (sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
            inteval = 30
    else:
        assert (0), "too many arguments"


    if (len(sys.argv) == 2):
        tracker = kcftracker.KCFTracker(True, False, True)  # hog, fixed_window, multiscale
        success, img = cap.read()
        bbox = cv2.selectROI("Tracking",img, False,False)
        print(bbox,[bbox])
        tracker.init([bbox[0],bbox[1],bbox[2],bbox[3]],img)
        font = cv2.FONT_HERSHEY_SIMPLEX

        #####
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('./data/outtest.mp4', fmt, frame_rate, size)
        #####

        while True:
            timer = cv2.getTickCount()
            success, img = cap.read()
            if success == False:
                break
            bbox,cx_,cy_, value = tracker.update(img)

            ###
            cx_ = bbox[0] + bbox[2] / 2.
            cy_ = bbox[1] + bbox[3] / 2.
            ###
            bbox = list(map(int, bbox))
            displacement.append([cx_,cy_,value])
            if img is None:
                break
            if success:
                drawBox(img, bbox)
            else:
                cv2.putText(img, "Tracking Lost", (15, 70), font, 0.5, (0, 0, 255), 2)
            # Frame rate per second
            fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
            cv2.putText(img, "fps" + str(int(fps)), (15, 30), font, 0.5, (255, 255, 255), 2)
            cv2.circle(img, (int(cx_), int(cy_)), 2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow("Tracking", img)

            ###
            writer.write(img)
            ###

            if cv2.waitKey(10) == 27:
                break
        ###
        writer.release()
        ###
        cap.release()
        cv2.destroyAllWindows()
    else:
        tracker = kcftracker.KCFTracker(True, False, True)  # hog, fixed_window, multiscale
        # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

        cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking', draw_boundingbox)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if (selectingObject):
                cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
            elif (initTracking):
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)

                tracker.init([ix, iy, w, h], frame)

                initTracking = False
                onTracking = True
            elif (onTracking):
                t0 = time()
                boundingbox,cx_,cy_, value = tracker.update(frame)
                displacement.append([cx_,cy_,value])
                t1 = time()

                boundingbox = list(map(int, boundingbox))
                cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                            (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1-t0
                cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

            cv2.imshow('tracking', frame)
            c = cv2.waitKey(inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

df = pd.DataFrame(displacement)
df.to_csv("data/set.csv")
