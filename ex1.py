#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""

import numpy as np
import cv2
import utils


def smoothImage(img):    
    """
    Given a coloured image apply a blur on the image, e.g. Gaussian blur
    """
    # TODO
    # img = cv2.imread('test.jpg')
    img = cv2.imread('test.jpg')
    img = cv2.GaussianBlur(img,(15,15),5)
    return img


def binarizeImage(img, thresh):
    """
    Given a coloured image and a threshold binarizes the image.
    Values below thresh are set to 0. All other values are set to 255
    """
    # TODO
    #img = cv2.imread('test.jpg',0) #into grey
    #img = cv2.imread(img,0) #into grey
    
    #img = cv2.imread('test.jpg',1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    return img


def processImage(img):
    """
    Given an coloured image applies the implemented smoothing and binarization.
    """
    # TODO
    img = cv2.imread('test.jpg')
    img = smoothImage(img)
    img = binarizeImage(img, 125)
    
    return img


def doSomething(img):
    """
    Given a coloured image apply any image manipulation. Be creative!
    """
    img = cv2.imread('test.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    masking=np.full((img1.shape[0], img1.shape[1]),0,dtype=np.uint8)
    for j in circles[0, :]:
        cv2.circle(masking, (j[0], j[1]), j[2], (255, 255, 255), -1)
    img = cv2.bitwise_or(img1, img1, mask=masking)

    return img

if __name__=="__main__":
    img = cv2.imread("test.jpg")
    utils.show(img)
    
    img1 = smoothImage(img)
    utils.show(img1)
    cv2.imwrite("result1.jpg", img1)
    
    img2 = binarizeImage(img, 125)
    utils.show(img2)
    cv2.imwrite("result2.jpg", img2)
   
    img3 = processImage(img)
    utils.show(img3)
    cv2.imwrite("result3.jpg", img3)
    
    img4 = doSomething(img)
    utils.show(img4)
    cv2.imwrite("result4.jpg", img4)
