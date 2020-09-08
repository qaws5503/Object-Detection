# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:27:39 2019

@author: Jimmy
"""
import mmcv
from mmdet.apis import inference_detector, show_result, init_detector
import numpy as np
import cv2
import pycocotools.mask as maskUtils

# cfg = mmcv.Config.fromfile('C:/Users/ASUS/Desktop/mmdetection/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py')
# model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
# checkpoint_file = load_checkpoint(model, 'C:/Users/ASUS/Desktop/ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth', map_location='cpu')
agrs = {'mode':'mask','color':1}
score_thr = 0.8
cfg = 'C:/Users/ASUS/Desktop/mmdetection/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'
checkpoint_file = 'C:/Users/ASUS/Desktop/ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'


# build the model from a config file and a checkpoint file
model = init_detector(cfg, checkpoint_file)

# test a single image and show the results
img = 'C:/Users/ASUS/Desktop/things/thesis/source_image2.png'
result = inference_detector(model, img)
result = (result[0], result[1][0])
if agrs['mode'] == 'all':
    show_result(img, result, model.CLASSES, score_thr=score_thr)

if agrs['mode'] == 'mask':
    image = cv2.imread(img)
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
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(image, image, mask=visMask)
            if agrs['color'] == 1:
                instance = cv2.add(instance,np.repeat(~visMask[:, :, np.newaxis], 3, axis=2))
            cv2.imshow('Class: '+str(model.CLASSES[labels[i]])+', Score: '+str(round(bboxes[i, -1],2))
                       ,instance)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

