from cv2 import dilate
from unet import UNET
import torch
from dataset import CMADataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import *
import numpy as np
import cv2
import json
from glob import glob
from utils_show import imshow_r, to_rgb
from PIL import Image
import pandas as pd

def save_checkpoint(model, optimizer, filename="./temp/my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    print(f"=> Saving checkpoint to {filename}")
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_path, model):
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)    
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_img_dir,
    train_label_dir,
    train_mask_dir,
    train_desc,
    val_img_dir,
    val_label_dir,
    val_mask_dir,
    val_desc,
    legend_type,
    batch_size,    
    num_workers=4,    
    pin_memory=True,
    num_samples = None,
    do_aug = False,
    persistent_workers = False,
    
    
):

    if legend_type == 'poly':
        # # samples with max number of polygon maps. 
        # train_samples = ['OK_250K.tif', 'DC_Wash_West.tif', 'CA_SanJose.tif', 'AK_Noatak.tif', 'CO_Bailey.tif']
        # test_samples = ['CA_Redding.tif', 'CA_InyoMtns.tif', 'ID_LakeWalcott.tif', 'CA_NV_LasVegas.tif', 'AZ_Flagstaff.tif']

        # samples with least number of polygon maps
        train_samples = ['MA_Nashua.tif', 'NE_PlatteR_2005a.tif', 'NE_PlatteR_2005b_basemap.tif', 'AK_LookoutRidge.tif', 'AK_PointLay.tif']
        test_samples = ['OR_Buxton.tif', 'AR_Jasper.tif', 'CA_SanSimeon.tif', 'ID_basement.tif', 'AK_Utukok.tif']
    else:
        train_samples = 15
        test_samples = 15

    transform = transforms.Compose([
        # resize
        transforms.RandomHorizontalFlip,
        transforms.RandomAffine,
        transforms.ColorJitter,
        transforms.GaussianBlur
        # normalize
    ])

    train_ds = CMADataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        mask_dir=train_mask_dir,
        input_desc=train_desc,
        num_samples=num_samples,
        legend_type=legend_type,
        do_aug=do_aug,
        transforms = transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        persistent_workers=persistent_workers,
        drop_last=True
    )

    # A mini train dataset to have a stable accuracy check while training.
    # Using num_samples param to achieve this. Get metrics on a fixed set of full images.
    train_ds_mini = CMADataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        mask_dir=train_mask_dir,
        input_desc=train_desc,        
        num_samples=train_samples,
        legend_type=legend_type,
        do_aug=do_aug
    )
    train_loader_mini = DataLoader(
        train_ds_mini,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        persistent_workers=persistent_workers,
        drop_last=True
    )

    val_ds = CMADataset(
        image_dir=val_img_dir,
        label_dir=val_label_dir,
        mask_dir=val_mask_dir,
        input_desc=val_desc,
        num_samples=test_samples,
        legend_type=legend_type,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        persistent_workers=persistent_workers,
        drop_last=True
    )

    return train_loader, val_loader, train_loader_mini

def check_accuracy(loader, model, device="cuda", num_batches = 50):

    
    model.eval()
    if num_batches == 'all':
        num_batches = len(loader)

    batch_dice_scores = []
    with torch.no_grad():
        for batch_ix, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float)
            y = y.to(device).unsqueeze(1)
            if isinstance(model, UNET):
                preds = model(x)
            else:
                preds = model(x)['out']
            preds = (preds > 0.5).float()
            batch_dice_scores.append((2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            ))
            if batch_ix == num_batches:
                break

    dice_avg = torch.mean(torch.tensor(batch_dice_scores))
    dice_std = torch.std(torch.tensor(batch_dice_scores))
    dice_median = torch.median(torch.tensor(batch_dice_scores))

    print(f"Dice score average: {dice_avg}")
    print(f"Dice score std : {dice_std}")
    print(f"Dice score median : {dice_median}")
    
    model.train()
    return dice_avg, dice_std, dice_median

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    # set model to evaluation mode, disbaling training mode
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            if isinstance(model, UNET):
                preds = model(x)
            else:
                preds = model(x)['out']
            preds = (preds > 0.5).float()
        overlays = draw_contours(x, preds, y)
        cv2.imwrite(f"{folder}/pred_{idx}_{EXP_NAME}.png", overlays)
        if idx == 15:
            break
    # set model back to training mode
    model.train()
    return 

def merge_images(image_batch, size):
    h,w = image_batch.shape[1], image_batch.shape[2]
    c = image_batch.shape[3]
    img = np.zeros((int(h*size[0]), w*size[1], c), dtype=float)
    for idx, im in enumerate(image_batch):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im
    return img

def draw_contours(img_batch, pred_batch, target_batch):
    '''
    Given batch of img, pred and target, draw contours to visualize the results
    '''
    img_batch = (img_batch.detach().cpu().numpy()*255).astype('uint8')
    pred_batch = (pred_batch.detach().cpu().numpy()*255).astype('uint8')
    target_batch = (target_batch.detach().cpu().numpy()*255).astype('uint8')
    overlays = []
    for imagex, pred, target in zip(img_batch, pred_batch, target_batch):

        image = np.moveaxis(imagex[:3,:,:], 0, -1)

        # Unable to draw contours on the image from dataloader directly.
        # Saving and loading the image somehow works
        tm_save_path = 'temp.png'
        cv2.imwrite(tm_save_path, image)
        image = cv2.imread(tm_save_path)

        legend = np.moveaxis(image[3:,:,:], 0, -1)

        pred_contours = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, pred_contours, -1, (0, 0, 255), 2)

        target_contours = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, target_contours, -1, (0, 255, 0), 2)

        image = image.astype(float)
        image /= 255.0
        overlays.append(image)

    os.remove(tm_save_path)
    overlays = np.array(overlays)
    im_merged = merge_images(overlays, [img_batch.shape[0]//8,8])

    return im_merged*255

def load_legend_data(img_name):
    '''
    Load legend data as a dict
    '''
    json_path = os.path.join(CHALLENGE_INP_DIR, 'training', img_name.replace('.tif', '.json'))
    with open(json_path) as f:
        legend_data = json.load(f)
    return legend_data

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def preprocess_points(points):
    '''Ensure that we have top-left and bottom-right coordinates of the legend'''
    # Get the outrmost points if coordinates of a polygon are given
    if len(points) > 2:
        points = bounding_box(points)

    # Sort points by x-axis
    points = sorted(points, key=lambda x : x[0])

    # Swap y point axis if needed
    # [0, 1        [1, 0]
    #  1, 0]   --> [0, 1]
    if points[0][1] > points[1][1]:
        points[0][1], points[1][1] = points[1][1], points[0][1]

    points = [(int(point[0]), int(point[1]))for point in points]
    return points

def get_only_object(img, mask, back_img, debug = False):
    '''
    foreground -> area of interest (predicitons + ground truth)
    from the input image, show only foreground in RGB, everything else in grayscale
    '''
    fg = cv2.bitwise_or(img, img, mask=mask)
    mask_inv = 1 - mask
    fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
    final = cv2.bitwise_or(fg, fg_back_inv)
    if debug:
        imshow_r('mask', mask, True)
        imshow_r('fg', fg)
        imshow_r('mask_inv', mask_inv)
        imshow_r('fg_back_inv', fg_back_inv)
        imshow_r('final', final, True)
    return final

def draw_contours_big(img_path, pred_path, target_path = None, debug = False):
    
    if isinstance(img_path, np.ndarray):
        img = img_path
    else:
        img = cv2.imread(img_path)
    img_grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img_grey = to_rgb(img_grey)
    if isinstance(pred_path, np.ndarray):
        pred = pred_path
    else:
        pred = cv2.imread(pred_path, 0)
    # print(np.unique(pred, return_counts=True))

    if target_path and os.path.exists(target_path):
        target = cv2.imread(target_path, 0)
        pred_or_target_mask = cv2.bitwise_or(pred, target)
        img = get_only_object(img, pred_or_target_mask, img_grey, False)
    else:
        img = get_only_object(img, pred, img_grey, False)

    if target_path and os.path.exists(target_path):
        target_contours = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(img, target_contours, -1, (0, 255, 0), 40)

    pred_contours = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(img, pred_contours, -1, (0, 0, 255), 20)

    if debug:
        imshow_r(os.path.basename(target_path), img, True)
        cv2.destroyAllWindows()
    return img

if __name__ == '__main__':

    # Overlay predictions and ground truth
    step = 'validation'
    csv_path = os.path.join(TILED_INP_DIR, INFO_DIR, f"challenge_{step}_files.csv")
    df = pd.read_csv(csv_path)

    inp_dir = os.path.join(CHALLENGE_INP_DIR, 'training' if step == 'testing' else step)
    pred_dir = os.path.join(RESULTS_DIR, step)
    # pred_dir = 'gaussiansolutionsteam'

    for ind, row in df.iterrows():

        save_path = os.path.join(RESULTS_CNTS_DIR, step, row['mask_fname'].replace('.tif', '.png'))
        if os.path.exists(save_path):
            continue

        if '_pt' not in row['mask_fname']:
            continue

        img_path = os.path.join(inp_dir, row['inp_fname'])
        pred_path = os.path.join(pred_dir, row['mask_fname'])
        target_path = os.path.join(inp_dir, row['mask_fname'])
        
        # Prediction is not available
        if not os.path.exists(pred_path):
            print(pred_path)
            print('Prediction is not generated. Run inference to generate predictions')
            continue

        print(img_path, pred_path, target_path)
        img = draw_contours_big(img_path, pred_path, target_path, debug = False)

        imshow_r(row['mask_fname'], img, True)
        # cv2.imwrite(save_path, img)

    # # Test dilation
    # mask_paths = glob(os.path.join(TRAIN_MASK_DIR, '*.tif'))
    # for mask_path in mask_paths:
    #     if 'poly' in mask_path:
    #         continue
    #     if 'pt' in mask_path:
    #         continue
    #     mask = cv2.imread(mask_path, 0)
    #     # img = cv2.imread(mask_path.replace('masks', 'inputs'))
    #     # imshow_r('img', img, True)
    #     if np.sum(mask):
    #         dilate_mask(mask)
    #         break

