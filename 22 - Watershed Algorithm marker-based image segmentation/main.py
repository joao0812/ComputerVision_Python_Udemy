import os

import cv2

import pandas as pd
import numpy as np
import matplotlib as mlp


def main():
    legoFloor = cv2.imread('22 - Watershed Algorithm marker-based image segmentation\\assets\\legoFloor.jpg')
    legoFloor = legoFloor[::2, ::2]

    legoFloor_gray = cv2.cvtColor(legoFloor, cv2.COLOR_BGR2GRAY)
    legoFloor_gray_blur = cv2.GaussianBlur(legoFloor_gray, (3,3), 0)

    cv2.imshow('Lego IMG', legoFloor)
    cv2.imshow('Lego Gray', legoFloor_gray)
    cv2.imshow('Lego Blur', legoFloor_gray_blur)

    ret1, thresh1 = cv2.threshold(legoFloor_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret1, thresh2 = cv2.threshold(legoFloor_gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Thresholded 1', thresh1)
    cv2.imshow('Thresholded 2', thresh2)
    
    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # sure background area
    sure_bg1 = cv2.dilate(opening1, kernel, iterations=3)
    sure_bg2 = cv2.dilate(opening2, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform1 = cv2.distanceTransform(opening1, cv2.DIST_L2, 5)
    ret1, sure_fg1 = cv2.threshold(dist_transform1, 0.7* dist_transform1.max(), 255, 0)
    dist_transform2 = cv2.distanceTransform(opening2, cv2.DIST_L2, 5)
    ret2, sure_fg2 = cv2.threshold(dist_transform2, 0.7* dist_transform2.max(), 255, 0)

    cv2.imshow('Sure FG 1', sure_fg1)
    cv2.imshow('Sure BG 1', sure_bg1)
    cv2.imshow('Sure FG 2', sure_fg2)
    cv2.imshow('Sure BG 2', sure_bg2)
    
    cv2.waitKey(0)


if '__main__' == __name__:
    main()
