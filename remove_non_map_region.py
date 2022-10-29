'''
Script to filter out non-map region from the whole .tif file
to speed up the training and inference.
'''
from distutils.log import debug
import json
import cv2
from glob import glob
import imutils
import numpy as np

from pycocotools.coco import COCO
from requests import patch
from sklearn.feature_extraction import img_to_graph
from torch import is_deterministic_algorithms_warn_only_enabled, true_divide
from utils_show import imshow_r, to_rgb
from config import *

def downscale_and_save_tifs(debug = False):
    '''
    Downscale the image to 1024*1024 and save for labeling
    '''

    json_paths = []
    for dir, _, _ in os.walk(CHALLENGE_INP_DIR):
        json_paths.extend(glob(os.path.join(dir,"*.json")))
    img_paths = [json_path.replace('.json', '.tif') for json_path in json_paths]

    for ind, img_path in enumerate(img_paths):
        
        print(f"{ind+1}/{len(img_paths)}")
        img_name = os.path.basename(img_path).split('.')[0]
        save_path = os.path.join(DOWNSCALED_DATA_DIR, 'imgs', img_name + '.png')
        img = cv2.imread(img_path)
        
        if debug:
            print(img.shape)
            imshow_r('img_raw', img)
        img_small = cv2.resize(img, (1024, 1024))
        if debug:
            print(img_small.shape)
            imshow_r('img_small', img_small, True)

        cv2.imwrite(save_path, img_small)
        # break

def generate_json_file_for_ls():
    '''
    '''
    ls_json = []
    img_paths = glob(os.path.join(DOWNSCALED_DATA_DIR, 'imgs', '*.png'))
    for img_path in img_paths:
        ls_json.append({"data": {"image" : img_path}})

    json_save_path = os.path.join(DOWNSCALED_DATA_DIR, 'label_studio_import.json')
    with open(json_save_path, 'w') as fp:
        json.dump(ls_json, fp)

def validate_the_labeling():
    '''
    Check if every legend raster file is part of the labeled so called 'map-region'. 
    This is to ensure we don't miss any map region
    '''
    pass

def generate_masks(debug = False):
    '''Generate mask files as png from annotations'''
    json_path = os.path.join(DOWNSCALED_DATA_DIR, 'export_from_label_studio.json')
    coco_data = COCO(json_path)

    for imgId in coco_data.getImgIds():

        img = coco_data.loadImgs(imgId)[0]
        print(img)
        annIds = coco_data.getAnnIds(imgId)        
        mask = np.zeros((img['height'], img['width']), dtype='uint8')
        for ann in coco_data.loadAnns(annIds):
            mask += coco_data.annToMask(ann)

        if debug:
            imshow_r('mask', mask*255, True)
        
        assert(np.max(mask) == 1)
        save_path = os.path.join(DOWNSCALED_DATA_DIR, 'masks', os.path.basename(img['file_name']))
        cv2.imwrite(save_path, mask)
        
def upscale_masks_to_original_size(debug = False):
    '''Scale the segs back to original dimensions of the .tif files'''
    mask_paths = glob(os.path.join(DOWNSCALED_DATA_DIR, 'masks', '*.png'))
    for mask_path in mask_paths:
        img_name = os.path.basename(mask_path).split('.')[0] + '.tif'
        img_path = os.path.join(CHALLENGE_INP_DIR, 'training', img_name)
        if not os.path.exists(img_path):
            img_path = img_path.replace('training', 'validation')
        img = cv2.imread(img_path)

        mask = cv2.imread(mask_path)
        if debug:
            imshow_r('mask_downscaled', mask*255)

        mask_upscaled = cv2.resize(mask, (img.shape[:2]), interpolation=cv2.INTER_NEAREST)
        if debug:
            imshow_r('mask_upscaled', mask_upscaled*255, True)

        upscaled_save_path = mask_path.replace('masks', 'masks_upscaled')
        cv2.imwrite(upscaled_save_path, mask_upscaled)



if __name__ == '__main__':
    # AR_Buffalo_west.png has map area towards right of the legend. 
    # Our current logic of postprocessing would fail in that case.
    # downscale_and_save_tifs()
    # generate_json_file_for_ls()
    # generate_masks()
    upscale_masks_to_original_size()