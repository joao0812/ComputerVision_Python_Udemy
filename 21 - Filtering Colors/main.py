import os

import cv2

import pandas as pd
import numpy as np
import matplotlib as mlp


def main():
    lower = np.array([91,120,0])
    upper = np.array([160,200,255])

    carAndNature = cv2.imread('21 - Filtering Colors\\assets\\carAndNature.jpg')
    carAndNature = carAndNature[::5, ::5]
    cv2.imshow('Frame', carAndNature)

    carAndNature_hsv = cv2.cvtColor(carAndNature, cv2.COLOR_BGR2HSV)
    cv2.imshow('Frame HSV', carAndNature_hsv)

    mask = cv2.inRange(carAndNature_hsv, lower, upper)
    cv2.imshow('Mask', mask)

    res_bit = cv2.bitwise_and(carAndNature, carAndNature, mask=mask)

    cv2.imshow('Frame Res', res_bit)

    cv2.waitKey(0)


if '__main__' == __name__:
    main()
