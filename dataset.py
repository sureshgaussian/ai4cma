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
# from config import IMG_DIR, LABEL_DIR, MASK_DIR, TRAIN_DESC

class CMADataset(Dataset):
    def __init__(self, image_dir, label_dir, mask_dir, input_desc, num_samples, use_median_color = False) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        
        self.input_desc = input_desc

        self.use_median_color = use_median_color
        if self.use_median_color:
            self.load_legend_median_values()
            input_df = pd.read_csv(self.input_desc)
            input_df['stripped'] = input_df['tile_legend'].apply(lambda x : x.split('.')[0])
            input_df = input_df[input_df['stripped'].isin(list(self.legend_data.keys()))]
            input_df = input_df.reset_index(drop=True)
            self.input_df = input_df
        else:
            self.input_df = pd.read_csv(self.input_desc)

        if num_samples:
            sample_org_files = self.input_df['orig_file'].unique()[:num_samples]
            input_df = self.input_df[self.input_df['orig_file'].isin(sample_org_files)]
            input_df = input_df.reset_index(drop=True)
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

        image = np.array(Image.open(img_path).convert("RGB"))
        image = image/255.0

        #Preprocess label based on using the median value
        if self.use_median_color:
            rgb = self.legend_data[reqd_row["tile_legend"].split('.')[0]]

            # Check if the median value matches with the image patch
            rgbArray = np.zeros(image.shape, 'uint8')
            rgbArray[..., 0] = rgb[0]
            rgbArray[..., 1] = rgb[1]
            rgbArray[..., 2] = rgb[2]
            # cv2.imshow('rgb', rgbArray)

            # median_encoded = ((rgb[0] + 1) + (rgb[1]+1)*256 + (rgb[2]+1)*256*256)/256.0**3
            # print(median_encoded)
            # label = np.full(image.shape[:2], median_encoded)
            # cv2.imshow('label', label*255)
            label = rgbArray/255.0

        else:
            label = np.array(Image.open(label_path).convert("RGB"))
            label = label/255.0

        label_mask = np.array(Image.open(mask_path).convert("L"))

        #image = np.expand_dims(image, axis=-1)
        #label = np.expand_dims(label, axis=-1)
        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        input = np.dstack((image, label))

        # Scaling is done seperately for image and mask. Hence we dont need this. 
        # input = input/255.0

        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        input= np.moveaxis(input, -1, 0)
        # label_mask= np.moveaxis(label_mask, -1, 0)

        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        return input, label_mask

    def load_legend_median_values(self):
        legend_median_data_path = '/home/ravi/cma_challenge/legends_median_data.json'
        with open(legend_median_data_path, "r") as fp:
            legend_data = json.load(fp)
        self.legend_data = legend_data



class CMAInferenceDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_desc, num_samples, use_median_color = False) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        self.input_desc = input_desc

        self.use_median_color = use_median_color
        if self.use_median_color:
            self.load_legend_median_values()
            input_df = pd.read_csv(self.input_desc)
            # print("length of input_df: ", len(input_df))
            input_df['stripped'] = input_df['label_pattern_fname'].apply(lambda x : x.split('.')[0])
            # print("length of input_df: ", input_df['stripped'])
            input_df = input_df[input_df['stripped'].isin(list(self.legend_data.keys()))]
            # print("length of input_df: ", len(input_df))
            input_df = input_df.reset_index(drop=True)
            self.input_df = input_df
            #print("length of self.input_df: ", len(self.input_df))
            assert(len(self.input_df) > 0)
        else:
            self.input_df = pd.read_csv(self.input_desc)

        if num_samples:
            sample_org_files = self.input_df['inp_file_name'].unique()[:num_samples]
            input_df = self.input_df[self.input_df['inp_file_name'].isin(sample_org_files)]
            input_df = input_df.reset_index(drop=True)
            self.input_df = input_df

        #print('Non empty label distribution : ', self.input_df['empty_tile'].value_counts())


    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self,index):

        # print(len(self.input_df))
        assert (index < len(self.input_df))
        reqd_row = self.input_df.loc[index]
        img_path = os.path.join(self.image_dir,reqd_row["in_tile"])
        label_path = os.path.join(self.label_dir,reqd_row["label_pattern_fname"])
        mask_tile_name = reqd_row["mask_tile_name"]

        image = np.array(Image.open(img_path).convert("RGB"))
        image = image/255.0

        #Preprocess label based on using the median value
        if self.use_median_color:
            rgb = self.legend_data[reqd_row["label_pattern_fname"].split('.')[0]]
            if len(rgb) == 1:
                trgb = rgb*3
                rgb = trgb 
                assert(len(rgb) == 3)

            # Check if the median value matches with the image patch
            rgbArray = np.zeros(image.shape, 'uint8')
            rgbArray[..., 0] = rgb[0]
            rgbArray[..., 1] = rgb[1]
            rgbArray[..., 2] = rgb[2]
            # cv2.imshow('rgb', rgbArray)

            # median_encoded = ((rgb[0] + 1) + (rgb[1]+1)*256 + (rgb[2]+1)*256*256)/256.0**3
            # print(median_encoded)
            # label = np.full(image.shape[:2], median_encoded)
            # cv2.imshow('label', label*255)
            label = rgbArray/255.0

        else:
            label = np.array(Image.open(label_path).convert("RGB"))
            label = label/255.0

        #image = np.expand_dims(image, axis=-1)
        #label = np.expand_dims(label, axis=-1)
        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        input = np.dstack((image, label))

        # Scaling is done seperately for image and mask. Hence we dont need this. 
        # input = input/255.0

        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        input= np.moveaxis(input, -1, 0)
        # label_mask= np.moveaxis(label_mask, -1, 0)

        # cv2.imshow('test', cv2.hconcat([image, label, np.stack((label_mask*255,)*3, axis=-1)]))
        return input, mask_tile_name

    def load_legend_median_values(self):
        #legend_median_data_path = '../data/all_legends_median_data.json'
        legend_median_data_path = '../eda/everything_legends_median_data.json'

        with open(legend_median_data_path, "r") as fp:
            legend_data = json.load(fp)
        self.legend_data = legend_data        


def test_cmadataset():
    ds = CMADataset(IMG_DIR, LABEL_DIR, MASK_DIR, TRAIN_DESC, num_samples = 2, use_median_color = True)
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
    ds = CMAInferenceDataset(input_dir, label_dir, validation_desc, num_samples = None, use_median_color = True)
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

