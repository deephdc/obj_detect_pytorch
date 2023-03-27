# -*- coding: utf-8 -*-
# import project config.py
"""
   Module to prepare the dataset
"""
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import obj_detect_pytorch.config as cfg
import os
import numpy as np
import torch
from PIL import Image
import subprocess

#Loads Dataset from  Nextcloud.
def download_dataset():
    try:
        images_path = os.path.join(cfg.DATASET_DIR, "Images") 
        masks_path = os.path.join(cfg.DATASET_DIR, "Masks")
        
        if not os.path.exists(images_path) and not os.path.exists(masks_path) :
            print('[INFO] No data found, downloading data...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', cfg.REMOTE_IMG_DATA_DIR, images_path])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            command = (['rclone', 'copy', '--progress', cfg.REMOTE_MASK_DATA_DIR , masks_path])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            print('[INFO] Finished.')
        else:
            print("[INFO] Images and masks folders already exist.")
            
    except OSError as e:
        output, error = None, e
    
#Creates a class of the dataset.
class Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join("obj_detect_pytorch/obj_detect_pytorch/dataset/", "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join("obj_detect_pytorch/obj_detect_pytorch/dataset/", "Masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join("obj_detect_pytorch/obj_detect_pytorch/dataset/", "Images", self.imgs[idx])
        mask_path = os.path.join("obj_detect_pytorch/obj_detect_pytorch/dataset/", "Masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
