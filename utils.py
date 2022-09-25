from ctypes import c_void_p
import torch
import torchvision
from dataset import CMADataset
from torch.utils.data import DataLoader
from config import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="./temp/my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(CHEKPOINT_PATH)    
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
    batch_size,
    num_workers=4,
    pin_memory=True,
    num_samples = None,
    use_median_color = False
):
    train_ds = CMADataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        mask_dir=train_mask_dir,
        input_desc=train_desc,
        num_samples=num_samples,
        use_median_color=use_median_color
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    # how is this possible?
    val_ds = CMADataset(
        image_dir=val_img_dir,
        label_dir=val_label_dir,
        mask_dir=val_mask_dir,
        input_desc=val_desc,
        num_samples=num_samples,
        use_median_color=use_median_color
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x)['out'])
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    # set model to evaluation mode, disbaling training mode
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)['out'])
            preds = (preds > 0.5).float()
        overlays = draw_contours(x, preds, y)
        cv2.imwrite(f"{folder}/pred_{idx}_{EXP_NAME}.png", overlays)
        # torchvision.utils.save_image(
        #     overlays, f"{folder}/pred_{idx}_{EXP_NAME}.png"
        # )
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}_{EXP_NAME}.png"
        # )
        # torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/{idx}.png")
        if idx == 5:
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
    img_batch = ((img_batch*255).detach().cpu().numpy()).astype('uint8')
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

    overlays = np.array(overlays)
    im_merged = merge_images(overlays, [2,8])

    return im_merged*255