from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from glob import glob
import os
import numpy as np
from pathlib import Path
import cv2
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imshow_r, to_rgb
import PIL
from configs_map import *

class MapDataset(Dataset):
    def __init__(self, data_root, step, do_aug = False) -> None:
        super().__init__()
        self.data_root = data_root
        self.step = step
        self.load_req_img_names()
        self.load_paths()
        self.do_aug = do_aug
        self.debug = False
        
    def __len__(self):
        return len(self.img_paths)

    def load_req_img_names(self):
        json_paths = glob(os.path.join(CHALLENGE_INP_DIR, self.step, '*.json'))
        img_names = [Path(json_path).stem for json_path in json_paths]
        self.req_img_names = img_names

    def load_paths(self):

        img_paths_all = glob(os.path.join(self.data_root, 'imgs', '*.png'))
        mask_paths_all = glob(os.path.join(self.data_root, 'masks', '*.png'))
    
        self.img_paths = []
        self.mask_paths = []

        for img_path, mask_path in zip(img_paths_all, mask_paths_all):
            if Path(img_path).stem in self.req_img_names:
                self.img_paths.append(img_path)
                self.mask_paths.append(mask_path)

    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index]).convert("RGB")
        img_name = os.path.basename(self.img_paths[index])

        if self.step in ['training', 'validation']:
            mask = Image.open(self.mask_paths[index]).convert('L')
        else:
            mask = np.array([])

        if self.debug:
            print(os.path.basename(self.img_paths[index]))
            print(img.size, mask.shape)
            img.show()
            print(img.mode)

        # Augmentations
        if self.do_aug and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

            if self.debug:
                imshow_r(f'H flipped {img_name}', [img, to_rgb(mask)], True)
        
        if self.do_aug and random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

            if self.debug:
                imshow_r(f'v flipped {img_name}', [img, to_rgb(mask)], True)

        if self.do_aug and random.random() > 0.5:
            img = PIL.ImageOps.invert(img)

            if self.debug:
                imshow_r(f'Inverted {img_name}', [img, to_rgb(mask)], True)


        img = TF.to_tensor(img)
        mask = np.asarray(mask)
        # img = np.asarray(img.copy())

        if self.debug:
            img_d = Image.fromarray(img.cpu().detach().numpy())
            img_d.show()
            

        return img, mask.copy(), img_name