import torch
from torch.utils.data import DataLoader
from cProfile import label
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import csv
import cv2
import pandas as pd
import random
import json
from config import ROOT_PATH
import torchvision.transforms.functional as TF
import PIL.ImageOps

from utils_show import dilate_mask
from utils_show import imshow_r, to_rgb
# from config import IMG_DIR, LABEL_DIR, MASK_DIR, TRAIN_DESC

class CMADataset(Dataset):
    def __init__(self, image_dir, label_dir, mask_dir, input_desc, num_samples, legend_type = 'line', do_aug = False) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sped_dir = label_dir.replace('legends', 'sped_legends')
        self.mask_dir = mask_dir
        self.input_desc = input_desc
        self.legend_type = legend_type
        self.do_aug = do_aug
        self.debug = True
       
        input_df = pd.read_csv(self.input_desc)

        # Filter by 'poly' or 'line' or 'pt'
        input_df = input_df[input_df['tile_legend'].str.contains(legend_type)]

        if legend_type == 'poly':
                self.load_legend_median_values()            
                # Discard invalid legends (legends with zero area)
                input_df['stripped'] = input_df['tile_legend'].apply(lambda x : x.split('.')[0])
                input_df = input_df[input_df['stripped'].isin(list(self.legend_data.keys()))]
                input_df.drop(columns=['stripped'], inplace=True)
                input_df.reset_index(drop=True, inplace=True)

        # Empty tiles do not contribute in semantic segmentation
        input_df = input_df[input_df['empty_tile'] == True]
        input_df.reset_index(drop=True, inplace=True)

        if num_samples:
            sample_org_files = input_df['orig_file'].unique()[:num_samples]
            input_df = input_df[input_df['orig_file'].isin(sample_org_files)]
            input_df.reset_index(drop=True, inplace=True)

        self.input_df = input_df

        print('Non empty label distribution : ', self.input_df['empty_tile'].value_counts())


    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self,index):

        assert (index < len(self.input_df))
        reqd_row = self.input_df.loc[index]
        img_path = os.path.join(self.image_dir,reqd_row["tile_inp"])
        label_path = os.path.join(self.label_dir,reqd_row["tile_legend"])
        mask_path = os.path.join(self.mask_dir,reqd_row["tile_mask"])
        sped_label_path = label_path.replace('legends', 'sped_legends')

        image = Image.open(img_path).convert("RGB")

        # Get the label that is to be concatenated with input
        if self.legend_type == 'poly':
            rgb = self.legend_data[reqd_row["tile_legend"].split('.')[0]]
            label = Image.new("RGB", image.size, tuple(rgb))
        else:
            if os.path.exists(sped_label_path):
                label = Image.open(sped_label_path).convert("RGB")
            else: # TODO : this needs to go away. We are using till suresh creates sped label folder for test
                label = Image.open(label_path).convert("RGB")

        label_mask = Image.open(mask_path).convert("L")

        # Widen lines and points in the mask(dilate)
        if self.legend_type in ['pt', 'line']:
            label_mask = dilate_mask(label_mask)

        # Perform the augmentations
        # TODO : The augmentation block to be replaced with torch compositions
        if self.debug:
            imshow_r(os.path.basename(mask_path), [image, label, to_rgb(label_mask)], True)

        if self.do_aug and random.random() > 0.5:
            image = TF.hflip(image)
            label_mask = TF.hflip(label_mask)

            if self.debug:
                imshow_r(f'H flipped {os.path.basename(mask_path)}', [image, label, to_rgb(label_mask)], True)
        
        if self.do_aug and random.random() > 0.5:
            image = TF.vflip(image)
            label_mask = TF.vflip(label_mask)

            if self.debug:
                imshow_r(f'v flipped {os.path.basename(mask_path)}', [image, label, to_rgb(label_mask)], True)

        if self.do_aug and random.random() > 0.5:
            image = PIL.ImageOps.invert(image)
            label = PIL.ImageOps.invert(label)

            if self.debug:
                imshow_r(f'Inverted {os.path.basename(mask_path)}', [image, label, to_rgb(label_mask)], True)

        # Convert to tensor finally
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)
        label_mask = np.array(label_mask)

        # Concatenate image and label        
        input = torch.cat((image, label))

        return input, label_mask

    def load_legend_median_values(self):
        legend_median_data_path = '/home/ravi/cma_challenge/legends_median_data.json'
        legend_median_data_path = os.path.join(ROOT_PATH, 'eda/everything_legends_median_data.json')
        with open(legend_median_data_path, "r") as fp:
            legend_data = json.load(fp)
        self.legend_data = legend_data


class CMAInferenceDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_desc, num_samples, legend_type="line") -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir        
        self.input_desc = input_desc
        self.debug = False
        self.legend_type = legend_type
        
        input_df = pd.read_csv(self.input_desc)

        if legend_type == 'poly':
                self.load_legend_median_values()
                # print("length of input_df: ", len(input_df))
                input_df['stripped'] = input_df['label_pattern_fname'].apply(lambda x : x.split('.')[0])
                # print("length of input_df: ", input_df['stripped'])
                input_df = input_df[input_df['stripped'].isin(list(self.legend_data.keys()))]
                # print("length of input_df: ", len(input_df))
                input_df = input_df.reset_index(drop=True)
                self.input_df = input_df
                

        if num_samples:
            sample_org_files = self.input_df['inp_file_name'].unique()[:num_samples]
            input_df = self.input_df[self.input_df['inp_file_name'].isin(sample_org_files)]
            input_df = input_df.reset_index(drop=True)
        
        self.input_df = input_df

        print("length of self.input_df: ", len(self.input_df))
        assert(len(self.input_df) > 0)

        #print('Non empty label distribution : ', self.input_df['empty_tile'].value_counts())


    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self,index):

        # print(len(self.input_df))
        assert (index < len(self.input_df))
        reqd_row = self.input_df.loc[index]
        img_path = os.path.join(self.image_dir,reqd_row["in_tile"])
        label_path = os.path.join(self.image_dir,reqd_row["label_pattern_fname"])
        mask_tile_name = reqd_row["mask_tile_name"]

        image = Image.open(img_path).convert("RGB")

        # Get the label that is to be concatenated with input
        if self.legend_type == 'poly':
            rgb = self.legend_data[reqd_row["label_pattern_fname"].split('.')[0]]
            # print("rgb=", rgb, "image size = ", image.size)
            if len(rgb) == 1:
                rgb = 3*rgb
            label = Image.new("RGB", image.size, tuple(rgb))
        else:
            if os.path.exists(label_path):
                label = Image.open(label_path).convert("RGB")

        if self.debug:
            imshow_r(f'{index} Image, Label', [image, label], True)

        # how come no normalization?
        # print(f'max of image is {np.max(np.array(image))}')
        image = TF.to_tensor(image)
        # print(f'max of image is {np.max(np.array(image))}')
        label = TF.to_tensor(label)
        input = torch.cat((image, label))

        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        return input, mask_tile_name

    def load_legend_median_values(self):
        #legend_median_data_path = '../data/all_legends_median_data.json'
        legend_median_data_path = os.path.join(ROOT_PATH, 'eda/everything_legends_median_data.json')

        with open(legend_median_data_path, "r") as fp:
            legend_data = json.load(fp)
        self.legend_data = legend_data       


def test_cmadataset():
    ds = CMADataset(IMG_DIR, LABEL_DIR, MASK_DIR, TRAIN_DESC, num_samples = 2)
    for i in random.sample(range(100), 50):
        (x,y) = ds[i]
        print(x.shape, y.shape)
        image = np.moveaxis(x[:3,:,:], 0, -1)
        label = np.moveaxis(x[3:,:,:], 0, -1)

        # print(np.min(image), np.max(image))
        # print(np.min(label), np.max(label))
        cv2.imshow('img', image)
        # cv2.imshow('label', label*255)
        cv2.imshow('mask', y*255)
        # cv2.imshow('test', cv2.hconcat([image, label]))      
        # y = np.moveaxis(y, 0, -1)
        # cv2.imshow('label_mask', y*255)
        # label_mask = np.stack((y,)*3, axis=-1)
        # print(image.shape, label.shape, label_mask.shape)
        # print(max(image), max(label), max(label_mask))
        # cv2.imshow('test', cv2.hconcat([image, label, label_mask]))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
def test_cma_inference_dataset():
    input_dir = "../data/short_test/tiled_inp"
    label_dir = "../data/short_test/tiled_inp"
    validation_desc = "predict.csv"
    ds = CMAInferenceDataset(input_dir, label_dir, validation_desc, num_samples = None)
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    TILE_SIZE = 256
    IMAGE_HEIGHT = TILE_SIZE  # 1280 originally
    IMAGE_WIDTH = TILE_SIZE  # 1918 originally
    PIN_MEMORY = True
    val_loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
        )
    for x,y in val_loader:
        print(len(y ))
        break
    # for x in random.sample(range(100), 50):
    #     (x, mask_tile_name) = ds[i]
    #     print(x.shape, mask_tile_name)
    #     # image = np.moveaxis(x[:3,:,:], 0, -1)
        # label = np.moveaxis(x[3:,:,:], 0, -1)

        # print(np.min(image), np.max(image))
        # print(np.min(label), np.max(label))
        #cv2.imshow('img', image)
        # cv2.imshow('label', label*255)
        #cv2.imshow('mask', y*255)
        # cv2.imshow('test', cv2.hconcat([image, label]))      
        # y = np.moveaxis(y, 0, -1)
        # cv2.imshow('label_mask', y*255)
        # label_mask = np.stack((y,)*3, axis=-1)
        # print(image.shape, label.shape, label_mask.shape)
        # print(max(image), max(label), max(label_mask))
        # cv2.imshow('test', cv2.hconcat([image, label, label_mask]))
        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break

if __name__ == "__main__":
    test_cma_inference_dataset()    
    #test_cmadataset()

