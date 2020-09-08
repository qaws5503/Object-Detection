# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:43 2019

@author: Jimmy
"""

import cv2
import numpy as np
import time
import os
import mmcv
from mmdet.apis import inference_detector, init_detector
import pycocotools.mask as maskUtils

def Mainprocess(inputMat1,inputMat2,processedImageView1,Object):
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
    
    
    visMask = Object[0]
    IdentifyObject(imageSegmentation1,imageSegmentation2, visMask)

    
#    combineImage = CombineImageRealMethod(imageSegmentation1,imageSegmentation2,inputMat1,inputMat2)
#    cv2.imshow('Result',combineImage)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def ImageBasicProcessing(src):
    processedImage=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    return processedImage

def InputImage(inputImage):
    return cv2.imread('inputImage',0)

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

def CutRoi(Input, startX, startY, W, H):
    endX = startX + W
    endY = startY + H
    CutRoi = Input[startY:endY, startX:endX]
    return CutRoi

def IdentifyObject(matNearRegionWithContour, matFarRegionWithContour, visMask):
    score = []
    Nearimage = matNearRegionWithContour
    Farimage = matFarRegionWithContour
    
    NearInstance = cv2.bitwise_and(Nearimage, Nearimage, mask=visMask)
    FarInstance = cv2.bitwise_and(Farimage, Farimage, mask=visMask)
    
    
    cv2.imshow('FarRoi',FarInstance)
    cv2.imshow('NearRoi',NearInstance)
    
    
    NearInstance = Change2Nan(NearInstance, visMask)
    FarInstance = Change2Nan(FarInstance, visMask)
    
    
    
    
    for r in range(0,FarInstance.shape[0]):
        for c in range(0,FarInstance.shape[1]):
            nearPixelValueOnSegmantation = NearInstance[r,c]
            farPixelvalueOnSegmatation = FarInstance[r,c]
            if (nearPixelValueOnSegmantation[0] == nearPixelValueOnSegmantation[0] or
                farPixelvalueOnSegmatation[0] == farPixelvalueOnSegmatation[0]):
                if ((nearPixelValueOnSegmantation[0] == 0) and (nearPixelValueOnSegmantation[1] == 0) and (nearPixelValueOnSegmantation[2] == 0)):
                    score.append('near')
                elif(farPixelvalueOnSegmatation[0] == 0 and farPixelvalueOnSegmatation[1] == 0 and farPixelvalueOnSegmatation[2] == 0):
                    score.append('far')
                else:
                    score.append('neither')
    
    most_freq = 0
    for l in ['near','far','neither']:
        if most_freq<score.count(l):
            most_freq=score.count(l)
            last_score = l
    print('near count: ' + str(score.count('near')) + ' pixels')
    print('Far count: ' + str(score.count('far')) + ' pixels')
    print('neither count: ' + str(score.count('neither')) + ' pixels')
    return last_score

def Change2Nan(Input, visMask):
    Input = Input.astype(float)
    for row in range(visMask.shape[0]):
        for col in range(visMask.shape[1]):
            if visMask[row][col] == 0:
                Input[row][col]=np.nan
    return Input

def dectectObject(image):
    score_thr = 0.8
    cfg = 'C:/Users/ASUS/Desktop/mmdetection/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'
    checkpoint_file = 'C:/Users/ASUS/Desktop/ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'
    Object = []
    
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint_file)
    
    # test a single image and show the results
    result = inference_detector(model, image)
    result = (result[0], result[1][0])
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        
        for i in inds:
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            visMask = (mask * 255).astype("uint8")
            # instance = cv2.bitwise_and(image, image, mask=visMask)
            # cv2.imshow('Class: '+str(model.CLASSES[labels[i]])+', Score: '+str(round(bboxes[i, -1],2))
            #            ,instance)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            Object.append(visMask)
    return Object



processedImage1=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image1.png')
processedImage2=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image2.png')
processedImageView1 = processedImage1
processedImageView2 = processedImage2
 
inputMat1 = processedImage1
inputMat2 = processedImage2
imageBasicProcessing1 = ImageBasicProcessing(inputMat1)
imageBasicProcessing2 = ImageBasicProcessing(inputMat2)
Object = dectectObject(processedImage1)
#Mainprocess(inputMat1,inputMat2,processedImageView1)







# original one
# =============================================================================
# processedImage1=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image1.png')
# processedImage2=cv2.imread('C:/Users/ASUS/Desktop/things/thesis/source_image2.png')
# processedImageView1 = processedImage1
# processedImageView2 = processedImage2
# 
# inputMat1 = processedImage1
# inputMat2 = processedImage2
# imageBasicProcessing1 = ImageBasicProcessing(inputMat1)
# imageBasicProcessing2 = ImageBasicProcessing(inputMat2)
# Mainprocess(inputMat1,inputMat2,processedImageView1)
# =============================================================================
