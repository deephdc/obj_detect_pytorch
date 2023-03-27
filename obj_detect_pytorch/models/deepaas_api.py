# -*- coding: utf-8 -*-
"""
Model description
"""
import argparse
import logging
import pkg_resources
import os
import torch
import cv2
from time import time
from PIL import Image
from fpdf import FPDF

import torchvision
from torchvision import transforms, utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import obj_detect_pytorch.models.model_utils as mutils
import obj_detect_pytorch.models.create_resfiles as resfiles 
import obj_detect_pytorch.dataset.make_dataset as mdata
import obj_detect_pytorch.models.transform as T
import obj_detect_pytorch.config as cfg
from obj_detect_pytorch.models.engine import train_one_epoch, evaluate
import obj_detect_pytorch.models.utils as utils2

# conigure python logger
logger = logging.getLogger('__name__')
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s')
logger.setLevel(cfg.log_level)

def get_metadata():
    #Metadata of the model:
    models_names = mutils.get_models()
    module = __name__.split('.', 1)
    pkg = pkg_resources.get_distribution(module[0])  
    meta = {
        'Name': None,
        'Models': str(models_names),
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par+":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta

def warm():
    """
    This is called when the model is loaded, before the API is spawned. 
    If implemented, it should prepare the model for execution. This is useful 
    for loading it into memory, perform any kind of preliminary checks, etc.
    """

def get_train_args():
    return cfg.train_args
    

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #-model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # if no mask available:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    logger.debug(F"[api.get_model_instance_] num_classes = {num_classes}")
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    #-in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #-hidden_layer = 256
    # and replace the mask predictor with a new one
    #-model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #-                                                   hidden_layer,
    #-                                                   num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

###
# Uncomment the following two lines
# if you allow only authorized people to do training
###
#from flaat import Flaat
#flaat = Flaat()
#@flaat.login_required()

def train(**args):
    #download dataset if it doens't exist.
    mdata.download_dataset()
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Using device:', device)
    torch.cuda.empty_cache()
    
    #saving names of the classes
    class_name = args['class_names']
    classes = class_name.split(',')

    # number of classes
    num_classes = int(args['num_classes'])
    
    # check if the number of classes coincides with the number of names
    if len(classes)!= num_classes:
        print('[ERROR] The number of classes is not the same as the number of names given.')
        run_results = "Error."
        return run_results
    
    # use our dataset and defined transformations
    dataset = mdata.Dataset('Dataset', get_transform(train=True))
    dataset_test = mdata.Dataset('Dataset', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=8,
        collate_fn=utils2.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=utils2.collate_fn)

    #get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    #move model to the right device
    model.to(device)

    #construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=float(args['learning_rate']) ,
                                momentum= float(args['momentum']), weight_decay= float(args['weight_decay']))
    
    #and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size= int(args['step_size']),
                                                   gamma= float(args['gamma']))

    #let's train it for num_epochs
    num_epochs = int(args['num_epochs'])

    data_size = len(data_loader) - 1
    test_size = len(data_loader_test) - 1
    
    t0 = time()
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print('[INFO] Evaluating model...')
        evaluate(model, data_loader_test, device=device)
    t1 = time()
    loss = str(metrics.__getattr__('loss'))
    loss_classifier =  str(metrics.__getattr__('loss_classifier'))
    loss_box = str(metrics.__getattr__('loss_box_reg'))
    #-loss_mask = str(metrics.__getattr__('loss_mask'))
    # if masks not available:
    loss_mask = "NaN"

    train_results = mutils.format_train(loss, loss_classifier, loss_box, loss_mask, num_epochs, 
                                       t1-t0,data_size, test_size)

    print("[INFO] Training done.")
    
    #writing file with the classes
    nums = [cfg.MODEL_DIR, args['model_name']]
    cat_file = '{0}/categories_{1}.txt'.format(*nums) 
    with open(cat_file, 'w') as filehandle:
        for listitem in classes:
            listitem = listitem.lstrip()
            filehandle.write('%s\n' % listitem)   

    #saving model's parameters
    model_path = '{0}/{1}.pt'.format(*nums)
    torch.save(model.state_dict(), model_path)
    print("[INFO] Model saved locally.")
    
    if (args['upload_model'] == 'true'):
        #copy model weigths, classes to nextcloud.
        dest_dir = cfg.REMOTE_MODELS_DIR
        print("[INFO] Upload %s to %s" % (model_path, dest_dir))
    
        #uploading class names to nextcloud.
        mutils.upload_model(cat_file)
    
        #uploading weights to nextcloud.
        mutils.upload_model(model_path)
        print("[INFO] Model uploaded.")
    
    print("[INFO] Finished training.")
    return train_results

def get_predict_args():
    return cfg.predict_args
    
def predict_file(**args):
    message = 'Not implemented in the model (predict_file)'
    return message

 
def predict(**args): 
    #Download weight files and model from nextcloud if necessary.
    if (args['model_name'] == "COCO"):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        CATEGORIES = mutils.category_names()
    else:
        mutils.download_model(args['model_name'])
        #To get the masks just add pred_mask in the prediction and results section.
        nums = [cfg.MODEL_DIR, args['model_name']]
        model_path = '{0}/{1}.pt'.format(*nums)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            logger.debug(F"state_dict: {state_dict}")
            #-model = get_model_instance_segmentation(list(state_dict["roi_heads.mask_predictor.mask_fcn_logits.bias"].size())[0])
            # if masks not available
            model = get_model_instance_segmentation(list(state_dict["roi_heads.box_predictor.cls_score.bias"].size())[0])
            model.load_state_dict(state_dict)
            CATEGORIES = []
            # open file and read the content in a list
            nums = [cfg.MODEL_DIR, args['model_name']]
            cat_file = '{0}/categories_{1}.txt'.format(*nums)
            with open(cat_file, 'r') as filehandle:
                for line in filehandle:
                    currentPlace = line[:-1]
                    CATEGORIES.append(currentPlace)
        else:
            message = 'Model not found.'
            return message
           
    model.eval()
    
    #tranform to tensor.
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    
    try:
        #reading the image and saving it.
        threshold= float(args['threshold'])
        thefile= args['files']
        img1 = Image.open(thefile.filename) 
    except Exception:
                raise ValueError("Failed parsing the input values. Check if the inputs meet the conditions.")
        
    other_path = '{}/Input_image_patch.png'.format(cfg.DATA_DIR)
    img1.save(other_path)
    
    #prediction and results.
    img1 = transform(img1) 
    pred = model([img1]) 
    pred_class = [CATEGORIES[i] for i in list(pred[0]['labels'].numpy())] 
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
    pred_score = list(pred[0]['scores'].detach().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]   #Boxes.
        pred_class = pred_class[:pred_t+1]   #Name of the class.
        pred_score = pred_score[:pred_t+1]   #Prediction probability.
    except IndexError:
        pred_t = 'null'
        pred_boxes = 'null'
        pred_class = 'null'
        pred_score = 'null'
        
    if (pred_t!='null'):
        if(args['accept'] == 'image/png'):
            #PDF Format:    
            #Drawing the boxes around the objects in the images + putting text + probabilities. 
            img_cv = cv2.imread(other_path) # Read image with cv2
            try:
                for i in range(len(pred_boxes)):
                    cv2.rectangle(img_cv, pred_boxes[i][0], pred_boxes[i][1], color= (124,252,0) , 
                                  thickness= int(args['box_thickness']))  # Draws rectangle.
                    cv2.putText(img_cv,str(pred_class[i]) + " " + str(float("{0:.4f}".format(pred_score[i]))), pred_boxes[i][0],
                            cv2.FONT_HERSHEY_SIMPLEX, int(args['text_size']), (124,252,0),thickness= int(args['text_thickness'])) 
                class_path = '{}/Classification_map.png'.format(cfg.DATA_DIR)
                cv2.imwrite(class_path,img_cv)    
            
            except Exception:
                raise ValueError("Failed parsing the input values. Check if the inputs meet the conditions.")
                
            #Merge original image with the classified one.
            result_image = resfiles.merge_images()

            #Create the PDF file.
            result_pdf = resfiles.create_pdf(result_image, pred_boxes, pred_class, pred_score)
            
            message = open(class_path, 'rb')
        else:
            message = mutils.format_prediction(pred_boxes,pred_class, pred_score)  
    return message

def main():
    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'train':
        train(args)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),
                            help=val['help'])

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict, train')
    args = parser.parse_args()

    main()
