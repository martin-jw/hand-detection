import numpy as np
import cv2

import hand_detect as hd
import threading
import queue

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN()

bg_frame = None


def id(binary, out_queue):

    binary = cv2.resize(binary, (400, 300))

    drawing, finger_data, palm_point = hd.identify_binary_image(binary / 255, False)
    drawing = np.uint8(drawing * 255)
    drawing = cv2.cvtColor(drawing, cv2.COLOR_RGB2BGR)

    out_queue.put(drawing)


q = queue.Queue()
t = threading.Thread()
while (True):
    ret, cframe = cap.read()

    frame = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(cframe, (160, 80), (480, 400), (0, 0, 255), 1)

    if bg_frame is None:
        bg_frame = frame

    fg = cv2.subtract(frame, bg_frame)
    b = np.zeros(shape=fg.shape, dtype=np.uint8)
    b[np.where(fg > 15)] = 255

    if not t.is_alive():
        t = threading.Thread(target=id, args=(b, q))
        t.isDaemon = True
        t.start()

    try:
        d = q.get_nowait()
        cv2.imshow('Detected', d)
    except:
        pass

    cv2.imshow('Webcam Feed', cframe)
    cv2.imshow('Binary Image', b)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
