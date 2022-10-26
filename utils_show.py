import numpy as np
import cv2
import imutils

def imshow_r(wdw, img, stop=False):
    ''' Parsing cv2.imshow() '''
    if isinstance(img, list):
        height, width = sorted(img, key=lambda x: x.shape[0])[0].shape[:2]
        imgs = [cv2.resize(im, (width, height)) for im in img]
        img = np.hstack(tuple(imgs))

    cv2.imshow(wdw, imutils.resize(img, height=800))
    
    if stop:
        cv2.waitKey()

def to_rgb(image):
    ''' Convert image to RGB '''
    return cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

def to_grayscale(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)