from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from glob import glob
import os
import numpy as np

class MapDataset(Dataset):
    def __init__(self, data_root, is_train = False) -> None:
        super().__init__()
        self.data_root = data_root
        self.is_train = is_train
        self.load_paths()
        self.debug = False
        
    def __len__(self):
        return len(self.img_paths)

    def load_paths(self):
        img_paths = glob(os.path.join(self.data_root, 'imgs', '*.png'))
        mask_paths = glob(os.path.join(self.data_root, 'masks', '*.png'))
        split = int(0.8*len(img_paths))
        if self.is_train:
            self.img_paths = img_paths[:split]
            self.mask_paths = mask_paths[:split]
        else:
            self.img_paths = img_paths[split:]
            self.mask_paths = mask_paths[split:]


    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert('L')
        # mask = np.array(mask)
        mask = np.asarray(mask)

        # assert img.size == mask.size

        if self.debug:
            print(os.path.basename(self.img_paths[index]))
            print(img.size, mask.shape)
            # # img.show()
            # print(np.max(mask))

        img = TF.to_tensor(img)
        

        return img, mask