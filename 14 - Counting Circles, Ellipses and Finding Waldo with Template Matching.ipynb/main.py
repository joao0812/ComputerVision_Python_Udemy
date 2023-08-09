import os

import cv2

import pandas as pd
import numpy as np
import matplotlib as mlp



def main():
    place = cv2.imread('14 - Counting Circles, Ellipses and Finding Waldo with Template Matching.ipynb/assets/wallyPlace2.jpg')
    wally = cv2.imread('14 - Counting Circles, Ellipses and Finding Waldo with Template Matching.ipynb/assets/wally2.jpg')

    cv2.imshow('Place', place)
    cv2.imshow('Wally', wally)

    place_gray = cv2.cvtColor(place, cv2.COLOR_BGR2GRAY)
    wally_gray = cv2.cvtColor(wally, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Place', place_gray)
    cv2.imshow('Wally', wally_gray)

    result = cv2.matchTemplate(place_gray, wally_gray, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(result)
    print(min_val)
    print(max_val)
    print(min_loc)
    print(max_loc)

    wally_find = place.copy()
    cv2.rectangle(wally_find, max_loc, (max_loc[0]+50, max_loc[1]+50), (255,0,0), 5)

    cv2.imshow('Find Wally', wally_find)

    cv2.waitKey(0)





if '__main__' == __name__:
    main()