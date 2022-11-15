'''
Have the post processing as a seperate step just to ensure that we dont disturb the raw predictions.
'''
import os
from glob import glob
from config import *
import pandas as pd
import cv2
import numpy as np
from inference import convert_mask_to_raster_tif
from PIL import Image
from utils import draw_contours_big
from utils_show import imshow_r
from postprocessing import generate_legend_bboxes_masks, remove_false_positives_within_map


def remove_false_positives_outside_map(step):
    '''
    Final post processing step to apply for all types (poly, line, pt)
    '''
    raw_predictions_dir = os.path.join(RESULTS_DIR, step)
    target_dir = os.path.join(POSTP_OUTMAP_DIR, step)
    os.makedirs(target_dir, exist_ok=True)
    map_only_masks_dir = os.path.join(ROOT_PATH, 'downscaled_data/predictions_upscaled', step)
    inp_desc_path = os.path.join(TILED_INP_DIR, INFO_DIR, f'challenge_{step}_set.csv')

    df = pd.read_csv(inp_desc_path)

    for ind, row in df.iterrows():
        
        map_only_path = os.path.join(map_only_masks_dir, row['inp_fname'].replace('.tif', '.png'))
        raw_pred_path = os.path.join(raw_predictions_dir, row['mask_fname'])
        print(map_only_path, raw_pred_path)
        map_only_mask = cv2.imread(map_only_path)
        raw_pred_mask = cv2.imread(raw_pred_path)
        
        print(np.unique(map_only_mask))
        print(np.unique(raw_pred_mask))
        print(ind, map_only_mask.shape, raw_pred_mask.shape)

        res = np.multiply(map_only_mask, raw_pred_mask)

        # imshow_r('res', res, True)
        print(res.shape)
        print(np.unique(res))

        # Sanity check. All channels should have same values
        b,g,r = res[:,:,0], res[:,:,1], res[:,:,2]
        assert ((b==g).all() and (b==r).all())
        
        save_path = os.path.join(target_dir, os.path.basename(raw_pred_path).split('.')[0]+'.tif')
        image = Image.fromarray(res[:,:,0], 'L')
        image.save(save_path)
        input_path = os.path.join(CHALLENGE_INP_DIR, 'training' if step == 'testing' else step, row['inp_fname'])
        convert_mask_to_raster_tif(input_path, save_path)
        break

if __name__ == '__main__':

    step = 'testing'
    generate_legend_bboxes_masks(step)
    remove_false_positives_within_map(step)
    remove_false_positives_outside_map(step)