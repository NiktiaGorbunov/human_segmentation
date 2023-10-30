import cv2
import mediapipe as mp
import numpy as np
import time


mp_drawind = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# документация по MediaPipe Selfie Segmentation
# https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html


BG_COLOR = (0, 255, 196)
cap = cv2.VideoCapture(0)  # подключаем вебку
# cap = cv2.VideoCapture('datasets/f17a6060-6ced-4bd1-9886-8578cfbb864f.mp4')  # подключаем вебку
prevTime = 0
# for webcam input

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True

        condition = np.stack((results.segmentation_mask, ) *3, axis= -1) > 0.1

        # apply some background magic
        bg_image = cv2.imread('backgrounds/1.png')  # create a virtual background  640 х 480
        # bg_image = cv2.imread('backgrounds/2.png')  # create a virtual background 1280 х 720
        # bg_image = cv2.GaussianBlur(image, (55, 55), 0)  # blur our background

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

        output_image = np.where(condition, image, bg_image)  # return elements chosen from x or y depending on condition

        # get frame rate
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        cv2.putText(output_image, f'FPS" {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 255), 2)

        cv2.imshow('DIY Zoom Virtual Background', output_image)
        if cv2.waitKey(5) & 0xFF == 27: # esc для выхода
            break

    cap.release()

