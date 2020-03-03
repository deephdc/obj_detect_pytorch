# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#

import requests
from os import listdir
import obj_detect_pytorch.config as cfg
import subprocess
from os import path

def format_prediction(boxes, labels, probabilities):
    d = {
        "results": [],
    }
    
    if (boxes != 'null'):
        for i in range(len(boxes)):
            pred = {
                "label": labels[i],
                "probability": str(probabilities[i]),
                "rectangle Coodinates":{
                    "coords": [{"Coordinates 1": str(boxes[i][0])},
                               {"Coordinates 2": str(boxes[i][1])}],
                },
            }
            d["results"].append(pred)
    else:
        d["results"].append("No classes found with the given threshold. Reduce threshold.")
    
    return d

def format_train(loss, classifier_loss, box_loss, mask_loss, nepochs,
                 time_prepare, data_size, test_size):

    train_info = {
        "network": 'Faster R-CNN.',
        "loss": {
                    "total loss": loss,
                    "classifier loss":classifier_loss,
                    "box loss":box_loss,
                    "mask loss": mask_loss
                }, 
        "n epochs": nepochs,
        "train set (images)": data_size,
        "test set (images)": test_size,
        "total time":  time_prepare
    }

    return train_info

def get_models():
    models = ['COCO']
    for f in listdir(cfg.MODEL_DIR): 
        if f.endswith(".pt"):
            models.append(f[:-3])
    return models

def category_names():
    
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    return COCO_INSTANCE_CATEGORY_NAMES
    
def upload_model(model_path):
    try:      
        #from the container to "rshare" remote storage 
        command = (['rclone', 'copy', '--progress', model_path, cfg.REMOTE_MODELS_DIR])
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e
        
        
def download_model(name):
    try:
        nums = [cfg.MODEL_DIR, name]
        model_path = '{0}/{1}.pt'.format(*nums)
        cat_path = '{0}/categories_{1}.txt'.format(*nums)
        
        if not path.exists(model_path) or not path.exists(cat_path):
            remote_nums = [cfg.REMOTE_MODELS_DIR, name]
            remote_model_path = '{0}/{1}.pt'.format(*remote_nums)
            remote_cat_path = '{0}/categories_{1}.txt'.format(*remote_nums)
            print('[INFO] Model not found, downloading model...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', remote_model_path, cfg.MODEL_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            command = (['rclone', 'copy', '--progress', remote_cat_path, cfg.MODEL_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            print('[INFO] Finished.')
        else:
            print("[INFO] Model found.")
            
    except OSError as e:
        output, error = None, e
    