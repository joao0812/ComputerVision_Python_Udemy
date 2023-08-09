import os

import cv2

import pandas as pd
import numpy as np
import matplotlib as mlp


def main():
    eye_classifier = cv2.CascadeClassifier(
        '16 - Face and Eye Detection with Haar Cascade Classifiers.ipynb/haarCascate/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        eye_detected = eye_classifier.detectMultiScale(frame_gray, 1.2, 3)
        print(f'Olho: {eye_detected}')
        print(f'Olho: {len(eye_detected)}')
        if len(eye_detected) > 1:
            if eye_detected[0][1] - eye_detected[1][1] <= 5:
                color = (0,0,255)
            else:
                color = (0,255,255)
            for (x, y, w, h) in eye_detected:
                print(x, y)
                cv2.putText(frame, f'X: {y}' ,(x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color, 1)
                cv2.circle(frame, ((x+w//2), (y+h//2)), 5, color, -1)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if '__main__' == __name__:
    main()
