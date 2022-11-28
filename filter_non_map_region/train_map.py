import os
import torch.nn as nn
import torch.optim as optim
import torch
from utils_map import load_model, get_loader, check_accuracy, save_checkpoint
from configs_map import *

if __name__ == '__main__':

    train_loader = get_loader(step = 'training', do_aug=True, shuffle = True)
    val_loader = get_loader(step = 'validation')

    print(f"train : {len(train_loader)}, validation : {len(val_loader)}")

    model = load_model()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f'Getting the scaler')
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 10
    for epoch in range(num_epochs):

        running_loss = 0
        for i, datum in enumerate(train_loader):
            
            data, targets, img_names = datum
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
        print("Checking  accuracy")
        check_accuracy(train_loader, model, num_batches='all')
        print("Checking VALIDATION accuracy")
        check_accuracy(val_loader, model, num_batches='all')
        CHEKPOINT_PATH_epoch_path = os.path.join('checkpoints', f"mapmodel_epoch_{epoch:02d}.pth.tar")
        save_checkpoint(model, optimizer, CHEKPOINT_PATH_epoch_path)