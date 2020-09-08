# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:55:47 2019

@author: Jimmy
"""

import cv2

processedImage1=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image1.png')
processedImage2=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image2.png')


inputMat1 = processedImage1
inputMat2 = processedImage2

#   Convert to grayscale
outputMat=cv2.cvtColor(inputMat1, cv2.COLOR_RGBA2GRAY)
#   Compute dx and dy derivatives
dx = cv2.Sobel(outputMat, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(outputMat, cv2.CV_32F, 0, 1)
dx = cv2.convertScaleAbs(dx)
dy = cv2.convertScaleAbs(dy)
#   Compute gradient
#// Core.magnitude(dx, dy, inputMat)
processMat1_edgeStrenght = cv2.addWeighted(dx,0.5,dy,0.5,0)
"after gredient the data type change to float but threshold olny can have a uint8 input"
resultMat = cv2.adaptiveThreshold(processMat1_edgeStrenght, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,11)

#processMat2_edgeStrenght = CalculateMapStrength(inputMat2).astype('uint8')
#processMat2_edgeStrenght = FilterMapStrengthWithAdaptiveThresholding(processMat2_edgeStrenght)
