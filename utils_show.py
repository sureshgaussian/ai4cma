import numpy as np
import cv2
import imutils
from PIL import Image
import os
from env import SAVE_DEBUG_IMGS

def pil_to_opencv(img):
    if not isinstance(img, np.ndarray):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if np.max(img) == 1:
        img = img*255
    return img

def imshow_r(wdw, img_src, stop=False):
    img = img_src.copy()
    ''' Parsing cv2.imshow() '''
    if isinstance(img, list):
        img = [pil_to_opencv(im) for im in img]
        height, width = sorted(img, key=lambda x: x.shape[0])[0].shape[:2]
        imgs = [cv2.resize(im, (width, height)) for im in img]
        img = np.hstack(tuple(imgs))
    else:
        img = pil_to_opencv(img)

    if SAVE_DEBUG_IMGS:
        cv2.imwrite(os.path.join('debug_images', wdw + '.png'), img)
        return

    cv2.imshow(wdw, imutils.resize(img, width=1800))
    
    if stop:
        cv2.waitKey()

def to_rgb(image):
    ''' Convert image to RGB '''
    if np.max(image == 1):
        image = image*255
    return cv2.cvtColor(np.array(image).copy(), cv2.COLOR_GRAY2RGB)

def to_grayscale(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

def dilate_mask(mask, debug = False):
    '''Extend points and lines in the masks
       to make it easier to train segmentation networks. 
    '''
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if debug:
        imshow_r('raw_mask', mask*255, True)    
    coords = np.argwhere(mask == 1)
    circle_radius = 5
    for coord in coords:
        cv2.circle(mask, (coord[1], coord[0]), circle_radius, 1, -1)
    if debug:
        imshow_r('with circle', mask*255, True)
    mask = Image.fromarray(mask)
    return mask