# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""
import os
from os import path
from webargs import fields, validate

import logging

# logging level accross various scripts
# options: DEBUG(10), INFO(20), WARNING(30), ERROR(40), CRITICAL(50)
env_log_level = os.getenv('OBJ_DETECT_LOG_LEVEL', 'INFO')
log_level = getattr(logging, env_log_level.upper(), 20) # INFO = 20

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))

DATA_DIR = path.join(BASE_DIR,'data') # Location of output files

DATASET_DIR = path.join(BASE_DIR,'obj_detect_pytorch/dataset/') # Location of the dataset

MODEL_DIR = path.join(BASE_DIR,'models') # Location of model data

#Change this to your remote folder path
Obj_det_RemoteSpace = 'rshare:/Datasets/obj_detec_pytorch/'
obj_det_ImageDataDir = 'data/Images/'
obj_det_MaskDataDir = 'data/Masks/'

REMOTE_IMG_DATA_DIR = path.join(Obj_det_RemoteSpace, obj_det_ImageDataDir)
REMOTE_MASK_DATA_DIR = path.join(Obj_det_RemoteSpace, obj_det_MaskDataDir)

REMOTE_MODELS_DIR = path.join(Obj_det_RemoteSpace, 'models/')

train_args = {
        "model_name": fields.Str(
            required=True,  # force the user to define the value
            description= "Name of the model without blank spaces. If another model with the same name exists it will be overwritten."  # help string
        ),
        
        "num_classes": fields.Str(
            required = True,  
            description= "Number of classes in the dataset. Note: It must be #classes + 1 since background is needed. Integer."
        ),
        
        "class_names": fields.Str(
            required=True,  
            description= "Names of the classes in the dataset. A background class must exist. The names must be separated by a coma, e.g. background,class1,class2."  
        ),
        
        "num_epochs": fields.Str(
            required=False,
            missing= 1,
            description= "Number of training epochs for the SGD." 
        ),
        
        "learning_rate": fields.Str(
            required=False, 
            missing= 0.005, 
            description= "Learning rate."  
        ),
        
        "momentum": fields.Str(
            required=False,  
            missing= 0.9, 
            description= "Momentum factor. Default: 0. More information: https://pytorch.org/docs/stable/optim.html"  
        ),
        
        "weight_decay": fields.Str(
            required=False,  
            missing= 0.0005,  
            description= "Weight decay (L2 penalty). Default: 0." 
        ),
        
        "step_size": fields.Str(
            required=False,  
            missing= 3,  
            description= "Period of learning rate decay, must be an integer." 
        ),
        
        "gamma": fields.Str(
            required=False,  
            missing= 0.1,  
            description= "Multiplicative factor of learning rate decay. Default: 0.1." 
        ),
        
        "upload_model": fields.Str(
            required=False,  
            missing= False,  
            enum=[True, False],
            description= "Set to True if the model and class names should be uploaded to nextcloud." 
        ),
    }


predict_args = {
        "model_name": fields.Str(
            required=False,  # force the user to define the value
            missing="COCO",  # default value to use
            description= "Name of the model. To see the available models please run the get_metadata function."  # help string
        ),

        "files": fields.Field(
            description="Data file to perform inference on.",
            required=True,
            type="file",
            location="form"),

        "threshold": fields.Str(
            required=False, 
            missing= 0.8,  
            description="Threshold of probability (0.0 - 1.0). Shows the predictions above the threshold."  
        ),
        
        "box_thickness": fields.Str(
            required=False,
            missing= 2, 
            description="Thickness of the box in pixels (Positive number starting from 1)."  
        ),
        
        "text_size": fields.Str(
            required=False,  
            missing= 1 , 
            description="Size of the text in pixels (Positive number starting from 1, for no text value is 0)."  
        ),
        
        "text_thickness": fields.Str(
            required=False,  
            missing= 2,  
            description="Thickness of the text in pixels (Positive number starting from 1)."  
        ),
        
        "accept" : fields.Str(
            require=False,
            description="Returns an image or a json with the box coordinates.",
            missing='image/png',
            validate=validate.OneOf(['image/png', 'application/json'])),

     }