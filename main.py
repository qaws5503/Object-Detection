# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:43 2019

@author: Jimmy
"""

import cv2
import numpy as np

def ImageBasicProcessing(src):
    processedImage=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    return processedImage

def InputImage(inputImage):
    return cv2.imread('inputImage',0)


def Mainprocess(inputMat1,inputMat2,processedImageView1):
    appliedWaterShed1 = inputMat1
    appliedWaterShed2 = inputMat2
    processMat1_edgeStrenght = CalculateMapStrength(inputMat1).astype('uint8')
    processMat1_edgeStrenght = FilterMapStrengthWithAdaptiveThresholding(processMat1_edgeStrenght)
    processMat2_edgeStrenght = CalculateMapStrength(inputMat2).astype('uint8')
    processMat2_edgeStrenght = FilterMapStrengthWithAdaptiveThresholding(processMat2_edgeStrenght)


    inputImageSegmentation1 = inputMat1
    intputImageSegmentation2 = inputMat2
    inputImageSegmentation1=cv2.cvtColor(inputImageSegmentation1,cv2.COLOR_BGRA2BGR)
    intputImageSegmentation2=cv2.cvtColor(intputImageSegmentation2,cv2.COLOR_BGRA2BGR)
    imageSegmentation1 = ImageSegmentation(inputImageSegmentation1, processMat1_edgeStrenght, appliedWaterShed1)
    imageSegmentation2 = ImageSegmentation(intputImageSegmentation2, processMat2_edgeStrenght, appliedWaterShed2)

    combineImage = CombineImageRealMethod(imageSegmentation1,imageSegmentation2,inputMat1,inputMat2)
    cv2.imshow('Result',combineImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CalculateMapStrength(inputMat):
#   Convert to grayscale
    outputMat=cv2.cvtColor(inputMat, cv2.COLOR_RGBA2GRAY)
#   Compute dx and dy derivatives
    dx = cv2.Sobel(outputMat, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(outputMat, cv2.CV_32F, 0, 1)
    dx = cv2.convertScaleAbs(dx)
    dy = cv2.convertScaleAbs(dy)
#   Compute gradient
#// Core.magnitude(dx, dy, inputMat)
    resultMat = cv2.addWeighted(dx,0.5,dy,0.5,0)
#// Core.magnitude(dx, dy, outputMat)
    return resultMat

def FilterMapStrengthWithAdaptiveThresholding(inputMat):
    resultMat = cv2.adaptiveThreshold(inputMat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,11)
#   Last version: blockSize is 51 and C is 0
    return resultMat

def ImageSegmentation(inputMat, edgeMat, waterShedResultMat):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    edgeMat = cv2.morphologyEx(edgeMat, cv2.MORPH_TOPHAT , structuringElement )
    outputMat = edgeMat
    contours, hierachy = cv2.findContours(edgeMat,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outputMat = inputMat
    cv2.drawContours(outputMat,contours,-1, (0,0,0),-1)
    return outputMat

def CombineImageRealMethod(matNearRegionWithContour, matFarRegionWithContour, matNearInput, matFarInput):
    
    nearFocusedSegmantation = matNearRegionWithContour
    farFocusedSegmantation = matFarRegionWithContour
    nearOriginal = matNearInput
    farOriginal = matFarInput
    numOfRow = np.size(matNearInput,0)
    numOfCol = np.size(matNearInput,1)
    SFMat = nearFocusedSegmantation
    for r in range(0,numOfRow):
        for c in range(0,numOfCol):
            nearPixelValueOnSegmantation = nearFocusedSegmantation[r,c]
            farPixelvalueOnSegmatation = farFocusedSegmantation[r,c]
            nearPixelValueOnOriginal = nearOriginal[r,c]
            farPixelValueOnOriginal = farOriginal[r,c]
            if ((nearPixelValueOnSegmantation[0] == 0) and (nearPixelValueOnSegmantation[1] == 0) and (nearPixelValueOnSegmantation[2] == 0)):
                value0 = 1*nearPixelValueOnOriginal[0]
                value1 = 1*nearPixelValueOnOriginal[1]
                value2 = 1*nearPixelValueOnOriginal[2]
                SFMat[r,c] = (value0, value1, value2)
            elif(farPixelvalueOnSegmatation[0] == 0 and farPixelvalueOnSegmatation[1] == 0 and farPixelvalueOnSegmatation[2] == 0):
                value0 = 1*farPixelValueOnOriginal[0]
                value1 = 1*farPixelValueOnOriginal[1]
                value2 = 1*farPixelValueOnOriginal[2]
                SFMat[r,c] = (value0,value1,value2)
            else:
                value0 = 0.5*(int(nearPixelValueOnOriginal[0]) + int(farPixelValueOnOriginal[0]))
                value1 = 0.5*(int(nearPixelValueOnOriginal[1]) + int(farPixelValueOnOriginal[1]))
                value2 = 0.5*(int(nearPixelValueOnOriginal[2]) + int(farPixelValueOnOriginal[2]))
                SFMat[r,c] = (value0,value1,value2)
    return SFMat


processedImage1=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image1.png')
processedImage2=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image2.png')
processedImageView1 = processedImage1
processedImageView2 = processedImage2

inputMat1 = processedImage1
inputMat2 = processedImage2
imageBasicProcessing1 = ImageBasicProcessing(inputMat1)
imageBasicProcessing2 = ImageBasicProcessing(inputMat2)
Mainprocess(inputMat1,inputMat2,processedImageView1)