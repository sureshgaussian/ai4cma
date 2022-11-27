import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset_map import MapDataset
from torch.utils.data import DataLoader
from glob import glob
import os
from pathlib import Path
import json
import cv2
import numpy as np
from configs_map import *

def load_model():

    model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)

    if torch.cuda.is_available():
        model.cuda()

    return model

def load_checkpoint(checkpoint_path, model = None):

    if not model:
        model = load_model()

    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)    
    model.load_state_dict(checkpoint["state_dict"])

    return model

def check_accuracy(loader, model, num_batches = 50):

    model.eval()
    if num_batches == 'all':
        num_batches = len(loader)

    batch_dice_scores = []
    with torch.no_grad():
        for batch_ix, (x, y, _) in enumerate(loader):
            x = x.to("cuda", dtype=torch.float)
            y = y.to("cuda").unsqueeze(1)
            preds = model(x)['out']
            preds = (preds > 0.5).float()
            batch_dice_scores.append((2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            ))
            if batch_ix == num_batches:
                break
    
    dice_avg = torch.mean(torch.tensor(batch_dice_scores))
    dice_min = torch.min(torch.tensor(batch_dice_scores))
    dice_std = torch.std(torch.tensor(batch_dice_scores))
    dice_median = torch.median(torch.tensor(batch_dice_scores))

    print(f"Dice score average: {dice_avg}")
    print(f"Dicer score min : {dice_min}")
    print(f"Dice score std : {dice_std}")
    print(f"Dice score median : {dice_median}")
    
    model.train()
    return dice_avg, dice_std, dice_median

def get_loader(step, do_aug = False, shuffle = False):

    dataset = MapDataset(DOWNSCALED_DATA_PATH, step = step, do_aug=do_aug)

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=shuffle
    )
    
    return loader

def save_checkpoint(model, optimizer, filename="./temp/my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    print(f"=> Saving checkpoint to {filename}")
    torch.save(checkpoint, filename)


def restore_prediction_dimensions(step):
    '''
    Raw predictions are of size 1024x1024. 
    Upscale the predictions to the original image size
    '''

    inp_dir = os.path.join(CHALLENGE_INP_DIR, step)
    json_paths = glob(os.path.join(inp_dir, '*.json'))

    for json_path in json_paths:

        with open(json_path) as fp:
            json_data = json.load(fp)

        out_w = json_data['imageWidth']
        out_h = json_data['imageHeight']

        pred_name = Path(json_path).stem + '.png'
        pred_path = os.path.join(DOWNSCALED_DATA_PATH, 'predictions_raw', step, pred_name)

        print(pred_path)

        pred = cv2.imread(pred_path, 0)

        # Draw a convex hull around all the blobs.
        # This approach still has some problems dealing with false positives.
        # But works in most of the cases.
        pred_final = np.zeros(pred.shape, dtype='uint8')

        contours = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours):
            cont = np.vstack(contours)
            hull = cv2.convexHull(cont)
            uni_hull = []
            uni_hull.append(hull)
            
            cv2.drawContours(pred_final, uni_hull, -1, 1, -1)
        # cv2.imshow('pred', pred*255)
        # cv2.imshow('pred_final', pred_final*255)
        # cv2.waitKey(0)

        # Restore prediction to orignal dimensions
        pred_restored = cv2.resize(pred_final, (out_w, out_h), cv2.INTER_NEAREST)
        pred_out_path = os.path.join(DOWNSCALED_DATA_PATH, 'predictions_upscaled', step, pred_name)
        cv2.imwrite(pred_out_path, pred_restored)

        cv2.destroyAllWindows()
        # break

def generate_downscaled_data(step):
    
    json_paths = glob(os.path.join(CHALLENGE_INP_DIR, step, '*.json'))
    img_paths = [json_path.replace('.json', '.tif') for json_path in json_paths]

    for ind, img_path in enumerate(img_paths):

        print(f"{ind}/{len(img_paths)} {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024))
        save_name = Path(img_path).stem + '.png'
        save_path = os.path.join(DOWNSCALED_DATA_PATH, 'imgs', save_name)
        cv2.imwrite(save_path, img)

if __name__ == '__main__':
    pass


