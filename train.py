#from ND_Crossentropy import DisPenalizedCE
import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
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

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device = {DEVICE}')
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
TILE_SIZE = 256
IMAGE_HEIGHT = TILE_SIZE  # 1280 originally
IMAGE_WIDTH = TILE_SIZE  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = './temp/tiled_inputs'
LABEL_DIR = "./temp/tiled_labels"
MASK_DIR = "./temp/tiled_masks"
TRAIN_DESC = "train.csv"
VAL_DESC = "test.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    print(f'In train function')
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        #print(f'got the data')
        data = data.to(device=DEVICE, dtype=torch.float)
        #print(f'Sent data to device')
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    

    model = UNET(in_channels=6, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    #loss_fn = GDiceLossV2()
    #loss_fn = DisPenalizedCE()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    )

    print(f'Got the loaders')

    if LOAD_MODEL:
        load_checkpoint(torch.load("./temp/my_checkpoint.pth.tar"), model)
        print(f'Checking accuracy of the pre-trained model')
        check_accuracy(val_loader, model, device=DEVICE)

    print(f'Getting the scaler')
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="./temp/saved_images/", device=DEVICE
        )


def test_save_predictions():
    model = UNET(in_channels=6, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("./temp/my_checkpoint.pth.tar"), model)
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
    )
    
    save_predictions_as_imgs(
            val_loader, model, folder="./temp/saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
    #test_save_predictions()