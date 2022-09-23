from cProfile import label
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import csv
import pandas as pd

class CMADataset(Dataset):
    def __init__(self, image_dir, label_dir, mask_dir, input_desc) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir

        self.input_desc = input_desc

        self.input_df = pd.read_csv(self.input_desc)

    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self,index):

        assert (index < len(self.input_df))
        reqd_row = self.input_df.loc[index]
        img_path = os.path.join(self.image_dir,reqd_row["tile_inp"])
        label_path = os.path.join(self.label_dir,reqd_row["tile_legend"])
        mask_path = os.path.join(self.mask_dir,reqd_row["tile_mask"])

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("RGB"))
        label_mask = np.array(Image.open(mask_path).convert("L"))

        #image = np.expand_dims(image, axis=-1)
        #label = np.expand_dims(label, axis=-1)
        input = np.concatenate( (image, label), axis=-1)
        input = input/255.0

        #print(f'{input.shape}')
        return input, label_mask


if __name__ == "__main__":
    ds = CMADataset("./temp/tiled_inputs", "./temp/tiled_labels", "./temp/tiled_masks", "train.csv")
    (X,y) = ds[10]
    print(X.shape, y.shape)
