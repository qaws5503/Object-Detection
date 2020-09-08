# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:40:43 2019

@author: Jimmy
"""

import cv2
import numpy as np
import time
import os

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
    
    (startX, startY, W, H, visMask) = decodeObject(Object[1])
    IdentifyObject(imageSegmentation1,imageSegmentation2, startX, startY, W, H, visMask)

    
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

def IdentifyObject(matNearRegionWithContour, matFarRegionWithContour, startX, startY, W, H, visMask):
    score = []
    Nearimage = matNearRegionWithContour
    Farimage = matFarRegionWithContour
    FarRoi = CutRoi(Farimage, startX, startY, W, H)
    NearRoi = CutRoi(Nearimage, startX, startY, W, H)
    FarInstance = cv2.bitwise_and(FarRoi, FarRoi, mask=visMask)
    NearInstance = cv2.bitwise_and(NearRoi, NearRoi, mask=visMask)
    
    cv2.namedWindow("FarRoi",0)
#    cv2.resizeWindow("FarRoi", 700, 700)
    cv2.imshow('FarRoi',FarInstance)
    cv2.namedWindow("NearRoi",0)
#    cv2.resizeWindow("NearRoi", 700, 700)
    cv2.imshow('NearRoi',NearInstance)
    
    
    FarInstance = ChangeRoitoNan(FarInstance, visMask)
    NearInstance = ChangeRoitoNan(NearInstance, visMask)
    
    
    
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
    print('near count: ' + str(score.count('near')))
    print('Far count: ' + str(score.count('far')))
    print('neither count: ' + str(score.count('neither')))
    return last_score

def ChangeRoitoNan(Input, visMask):
    Input = Input.astype(float)
    for row in range(visMask.shape[0]):
        for col in range(visMask.shape[1]):
            if visMask[row][col] == 0:
                Input[row][col]=np.nan
    return Input

def dectectObject(image):
    args = {'confidence':0.5, 'threshold':0.3, 'visualize':1}
    
    # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = os.getcwd()+"/mask-rcnn/mask-rcnn-coco/object_detection_classes_coco.txt"
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    colorsPath = os.getcwd()+"/mask-rcnn/mask-rcnn-coco/colors.txt"
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")
    
    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.getcwd()+"/mask-rcnn/mask-rcnn-coco/frozen_inference_graph.pb"
    configPath = os.getcwd()+"/mask-rcnn/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    
    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    
    # load our input image and grab its spatial dimensions
#    image = cv2.imread(os.getcwd()+"/mask-rcnn/images/source3_1.png")
    (H, W) = image.shape[:2]
    
    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()
    
    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))
    
    result = []
    Object = []
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
    
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > args["confidence"]:
            # clone our original image so we can draw on it
            clone = image.copy()
    
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
    
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])
    
            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]
    
            # check to see if are going to visualize how to extract the
            # masked region itself
            """
            if args["visualize"] > 0:
            """
            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)
            
    
            # only show the object and let the other
            # part be empty
    
    
            img = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
            img.fill(0)
            img[startY:endY, startX:endX] = instance
            result.append(img)
            
            Object.append([startX, startY, boxW, boxH, visMask])
    return Object

def decodeObject(Object):
    startX = Object[0]
    startY = Object[1]
    boxW = Object[2]
    boxH = Object[3]
    visMask = Object[4]
    return startX, startY, boxW, boxH, visMask

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
