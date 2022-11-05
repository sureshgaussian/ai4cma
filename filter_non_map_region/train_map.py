from dataset_map import MapDataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.optim as optim
import torch

DOWNSCALED_DATA_PATH = '/home/suresh/challenges/ai4cma/downscaled_data'
BATCH_SIZE = 2
NUM_WORKERS = 1


def get_loaders():
    train_ds = MapDataset(DOWNSCALED_DATA_PATH)
    test_ds = MapDataset(DOWNSCALED_DATA_PATH, is_train=False)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    return train_loader, test_loader

def save_checkpoint(model, optimizer, filename="./temp/my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    print(f"=> Saving checkpoint to {filename}")
    torch.save(checkpoint, filename)

def check_accuracy(loader, model, num_batches = 50):

    model.eval()
    if num_batches == 'all':
        num_batches = len(loader)

    batch_dice_scores = []
    with torch.no_grad():
        for batch_ix, (x, y) in enumerate(loader):
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
    dice_std = torch.std(torch.tensor(batch_dice_scores))
    dice_median = torch.median(torch.tensor(batch_dice_scores))

    print(f"Dice score average: {dice_avg}")
    print(f"Dice score std : {dice_std}")
    print(f"Dice score median : {dice_median}")
    
    model.train()
    return dice_avg, dice_std, dice_median


if __name__ == '__main__':

    train_loader, test_loader = get_loaders()

    model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)

    if torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f'Getting the scaler')
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 10
    for epoch in range(num_epochs):

        running_loss = 0
        for i, datum in enumerate(train_loader):
            
            data, targets = datum
            data = data.to(device="cuda", dtype=torch.float)
            targets = targets.float().unsqueeze(1).to(device="cuda")
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(data)['out']
                loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"{i+1}/{len(train_loader)} : loss : {running_loss/(i+1)}")

        print(f"Epoch : {epoch}/{num_epochs} : loss : {running_loss/len(train_loader)}")
        check_accuracy(test_loader, model, num_batches='all')
        CHEKPOINT_PATH_epoch_path = f"mapmodel_epoch_{epoch:02d}.pth.tar"
        save_checkpoint(model, optimizer, CHEKPOINT_PATH_epoch_path)