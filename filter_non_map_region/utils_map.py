import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset_map import MapDataset
from torch.utils.data import DataLoader

DOWNSCALED_DATA_PATH = '/home/suresh/challenges/ai4cma/downscaled_data'

BATCH_SIZE = 2
NUM_WORKERS = 2

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

def get_loader(step, do_aug = False):

    dataset = MapDataset(DOWNSCALED_DATA_PATH, step = step, do_aug=do_aug)

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    return loader

def save_checkpoint(model, optimizer, filename="./temp/my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    print(f"=> Saving checkpoint to {filename}")
    torch.save(checkpoint, filename)