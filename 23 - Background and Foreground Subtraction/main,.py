# Our Setup, Import Libaries, Create our Imshow Function and Download our Images
import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

foreground_background = cv2.createBackgroundSubtractorKNN()
corridaVideo = cv2.VideoCapture('23 - Background and Foreground Subtraction\\assets\\corridaVideo.mp4')

ret2, frameCorrida = corridaVideo.read()
frameCorrida = frameCorrida[::2, ::2]
avarage = np.float32(frameCorrida)

while True:
    ret, frame = cap.read()
    ret2, frameCorrida = corridaVideo.read()
    frameCorrida = frameCorrida[::2, ::2]

    frameCorrida_hsv = cv2.cvtColor(frameCorrida, cv2.COLOR_BGR2HSV)

    if not ret:
        break

    foreground_mask = foreground_background.apply(frameCorrida)
    foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

    cv2.accumulateWeighted(frameCorrida, avarage, 0.01)
    bg = cv2.convertScaleAbs(avarage)

    foreground_mask_bg = foreground_background.apply(bg)
    foreground_mask_bg = cv2.cvtColor(foreground_mask_bg, cv2.COLOR_GRAY2BGR)

    foreground_mask_hsv = foreground_background.apply(frameCorrida_hsv)
    foreground_mask_hsv = cv2.cvtColor(foreground_mask_hsv, cv2.COLOR_GRAY2BGR)

    res = np.vstack([np.hstack([frameCorrida, foreground_mask]), np.hstack([bg, foreground_mask_bg]), np.hstack([frameCorrida_hsv, foreground_mask_hsv])])
    cv2.imshow('Frame', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

