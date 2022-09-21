from cProfile import label
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import csv
class CMADataset(Dataset):
    def __init__(self, image_dir, input_desc) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.input_desc = input_desc
        self.input_list =  []
        with open(input_desc) as inp:
            csvFile = csv.reader(inp)
            line_count = 0
            for row in csvFile:
                if line_count == 0:
                    print(f'Column names in input are: {", ".join(row)}')
                else:
                    self.input_list.append(row)
                line_count += 1
        print(f'Read {line_count} lines')

    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self,index):
        reqd_row = self.input_list[index]
        img_path = reqd_row[2]
        label_path = reqd_row[3]
        label_mask_path = reqd_row[4]
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("RGB"))
        label_mask = np.array(Image.open(label_mask_path).convert("L"))

        #image = np.expand_dims(image, axis=-1)
        #label = np.expand_dims(label, axis=-1)
        input = np.concatenate( (image, label), axis=-1)
        input = input/255.0

        #print(f'{input.shape}')
        return input, label_mask


if __name__ == "__main__":
    ds = CMADataset("./temp", "inputs.csv")
    (X,y) = ds[0]
    print(X.shape, y.shape)
