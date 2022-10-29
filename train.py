import argparse
import torch
import os
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
from config import *
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

from dice_loss import (
    GDiceLossV2
)

def train_fn(epoch_index, loader, model, optimizer, loss_fn, scaler):
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

        # backward
        # print('next')
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # update tqdm loop
        # loop.set_postfix(loss=loss.item())
        # Gather data and report
        if i % 100 == 9:
            running_loss += loss.item()
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    return last_loss


def main(args):

    TRAIN_IMG_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/inputs")
    TRAIN_LABEL_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/legends")
    TRAIN_MASK_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_training/masks")
    TRAIN_DESC = os.path.join(TILED_INP_DIR, args.dataset+"_training/info/balanced_tiles.csv")

    TEST_IMG_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/inputs")
    TEST_LABEL_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/legends")
    TEST_MASK_DIR = os.path.join(TILED_INP_DIR, args.dataset+"_testing/masks")
    TEST_DESC = os.path.join(TILED_INP_DIR, args.dataset+"_testing/info/balanced_tiles.csv")


    if MODEL_NAME == 'unet':
        model = UNET(in_channels=IN_CHANNELS, out_channels=1).to(DEVICE)
    else:
        model = deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        model.backbone.conv1 = nn.Conv2d(IN_CHANNELS, 64, 7, 2, 3, bias=False)

    if torch.cuda.is_available():
        model.cuda()

    # 
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    #loss_fn = GDiceLossV2()
    #loss_fn = DisPenalizedCE()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
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
        USE_MEDIAN_COLOR
    )

    print(f'Got the loaders')

    checkpoint_path = args.model_checkpoint_path
    if LOAD_MODEL:
        load_checkpoint(checkpoint_path, model)
        print(f'Checking accuracy of the pre-trained model')
        check_accuracy(val_loader, model, device=DEVICE)

    print(f'Getting the scaler')
    scaler = torch.cuda.amp.GradScaler()

    for epoch_index in range(NUM_EPOCHS):
        last_loss = train_fn(epoch_index, train_loader, model, optimizer, loss_fn, scaler)

        print(epoch_index, last_loss)

        # if epoch_index % 10 == 9:

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, checkpoint_path)

        # check accuracy
        check_accuracy(train_loader, model, device=DEVICE)
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder. Use train loader for dry runs
        if NUM_SAMPLES:
            save_predictions_as_imgs(
                train_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
            )
        else:
            save_predictions_as_imgs(
                val_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
            )
                

def test_save_predictions():
    model = UNET(in_channels=IN_CHANNELS, out_channels=1).to(DEVICE)
    load_checkpoint(CHEKPOINT_PATH, model)
    train_loader, val_loader = get_loaders(
        IMG_DIR,
        LABEL_DIR,
        MASK_DIR,
        TRAIN_DESC,
        IMG_DIR,
        LABEL_DIR,
        MASK_DIR,
        VAL_DESC,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        NUM_SAMPLES,
        USE_MEDIAN_COLOR
    )

    check_accuracy(val_loader, model, device=DEVICE)
    
    # save_predictions_as_imgs(
    #         val_loader, model, folder=SAVED_IMAGE_PATH, device=DEVICE
    #     )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training parser')
    parser.add_argument('-d', '--dataset', default='mini', help='which dataset [ mini, challenge]')
    parser.add_argument('-t', '--tile_size', default=TILE_SIZE, help='tile size INT')
    parser.add_argument('-m', '--model_checkpoint_path', default=CHEKPOINT_PATH, help='checkpoint path')
    args = parser.parse_args()
    main(args)
    # test_save_predictions()