# import the necessary packages
import numpy as np
import time
import cv2
import os


args = {'confidence':0.5, 'threshold':0.3, 'visualize':1, 'color':1}

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
image = cv2.imread(os.getcwd()+"/mask-rcnn/images/source3_1.png")
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
        
        if args["color"]==0:

            visMask = ~visMask
            visMask = np.dstack([visMask]*3)
            mask1 = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
            mask1.fill(255) # or img[:] = 255
            mask1[startY:endY, startX:endX] = visMask
            img = img|mask1
        
        # show the output image
#        cv2.imshow("Output", img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        Object.append([startX, startY, boxW, boxH, visMask])
        