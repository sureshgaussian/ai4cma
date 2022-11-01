import argparse
import torch
import os
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
from config import *
from torchvision.models.segmentation import deeplabv3_resnet101

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def train_fn(epoch_index, loader, test_loader, model, optimizer, loss_fn, scaler, train_loader_mini):
    # print(f'In train function')
    # loop = tqdm(loader)
    running_loss = 0.
    last_loss = 0.

    for i, datum in enumerate(loader):
        #print(f'got the data')
        data, targets = datum
        data = data.to(device=DEVICE, dtype=torch.float)
        #print(f'Sent data to device')
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        optimizer.zero_grad()
        # forward
        with torch.cuda.amp.autocast():
            if isinstance(model, UNET):
                predictions = model(data)
            else:
                predictions = model(data)['out']
            loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        # Print loss to console every 10th step
        if i % 10 == 9:
            running_loss += loss.item()
            last_loss = running_loss / 10 # loss per batch
            print(f"epoch {epoch_index}  batch {i + 1}/{len(loader)} loss: {last_loss}")
            running_loss = 0.
            
        # Write to tensorboard every 500th step
        if i % 500 == 499:

            global_step_index = epoch_index*len(loader) + i
            writer.add_scalar(f"train_loss", last_loss, global_step_index)

            train_dice_avg, train_dice_std, train_dice_median = check_accuracy(train_loader_mini, model, device=DEVICE, num_batches='all')
            writer.add_scalar(f"train_dice_avg", train_dice_avg, global_step_index)
            writer.add_scalar(f"train_dice_std", train_dice_std, global_step_index)
            writer.add_scalar(f"train_dice_median", train_dice_median, global_step_index)

            test_dice_avg, test_dice_std, test_dice_median = check_accuracy(test_loader, model, device=DEVICE, num_batches='all')
            writer.add_scalar(f"test_dice_avg", test_dice_avg, global_step_index)
            writer.add_scalar(f"test_dice_std", test_dice_std, global_step_index)
            writer.add_scalar(f"test_dice_median", test_dice_median, global_step_index)

            writer.flush()

        # Save checkpoint every 2k step or at the last step of the epoch
        if i % 2000 == 1999 or i == len(loader) - 1:
            CHEKPOINT_PATH_epoch_path = CHEKPOINT_PATH.replace('.pth.tar', f"_epoch_{epoch_index:02d}_step_{i:05d}.pth.tar")
            save_checkpoint(model, optimizer, CHEKPOINT_PATH_epoch_path)
            save_predictions_as_imgs(
                test_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
            )

    return last_loss


def main(args):

    POLY_TRAINING_CSV = "balanced_tiles.csv"
    TRAIN_IMG_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/inputs")
    TRAIN_LABEL_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/legends")
    TRAIN_MASK_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/masks")
    TRAIN_DESC = os.path.join(TILED_INP_DIR, args.dataset+"_training/info/"+POLY_TRAINING_CSV)

    TEST_IMG_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/inputs")
    TEST_LABEL_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/legends")
    TEST_MASK_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/masks")
    TEST_DESC = os.path.join(TILED_INP_DIR, args.dataset+"_testing/info/"+POLY_TRAINING_CSV)

    MODEL_NAME = 'deeplabv3'
    if MODEL_NAME == 'unet':
        model = UNET(in_channels=IN_CHANNELS, out_channels=1).to(DEVICE)
    else:
        model = deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        model.backbone.conv1 = nn.Conv2d(IN_CHANNELS, 64, 7, 2, 3, bias=False)

    if torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader, train_loader_mini = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        TRAIN_MASK_DIR,
        TRAIN_DESC,
        TEST_IMG_DIR,
        TEST_LABEL_DIR,
        TEST_MASK_DIR,
        TEST_DESC,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        NUM_SAMPLES,
    )

    print(f'Got the loaders')

    checkpoint_path = args.model_checkpoint_path
    if LOAD_MODEL:
        load_checkpoint(checkpoint_path, model)
        print(f'Checking accuracy of the pre-trained model')
        check_accuracy(test_loader, model, device=DEVICE)

    print(f'Getting the scaler')
    scaler = torch.cuda.amp.GradScaler()

    for epoch_index in range(NUM_EPOCHS):
        last_loss = train_fn(epoch_index, train_loader, test_loader, model, optimizer, loss_fn, scaler, train_loader_mini)
        writer.add_scalar("train_loss", last_loss, epoch_index)

        print(epoch_index, last_loss)

        # save checkpoint
        save_checkpoint(model, optimizer, checkpoint_path)

        # check accuracy
        print(f'Training metrics...')
        check_accuracy(train_loader, model, device=DEVICE)
        print(f'Testing metrics...')
        check_accuracy(test_loader, model, device=DEVICE)

        if NUM_SAMPLES:
            save_predictions_as_imgs(
                train_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
            )
        else:
            save_predictions_as_imgs(
                test_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
            )         

def test_save_predictions():
    model = deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
    model.backbone.conv1 = nn.Conv2d(IN_CHANNELS, 64, 7, 2, 3, bias=False)
    if torch.cuda.is_available():
        model.cuda()
    load_checkpoint(CHEKPOINT_PATH, model)
    _, test_loader, _ = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        TRAIN_MASK_DIR,
        TRAIN_DESC,
        TEST_IMG_DIR,
        TEST_LABEL_DIR,
        TEST_MASK_DIR,
        TEST_DESC,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        NUM_SAMPLES,
        PERSISTANT_WORKERS
    )

    check_accuracy(test_loader, model, device=DEVICE, num_batches=100)
    
    save_predictions_as_imgs(
            test_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training parser')
    parser.add_argument('-d', '--dataset', default='mini', help='which dataset [ mini, challenge]')
    parser.add_argument('-t', '--tile_size', default=TILE_SIZE, help='tile size INT')
    parser.add_argument('-m', '--model_checkpoint_path', default=CHEKPOINT_PATH, help='checkpoint path')
    args = parser.parse_args()
    main(args)
    # test_save_predictions()
