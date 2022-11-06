from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
from utils import get_encoded_loader
from config import *
import torch.optim as optim
from torchvision import models
from torchsummary import summary

deeplab = models.segmentation.deeplabv3_resnet101()
deeplab.cuda()
summary(deeplab, (3, 224, 224))
print(deeplab)

loader = get_encoded_loader()

model = DeepLabHead(2048, 1)
model.cuda()

summary(model, (2048, 1, 1))
print(model)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for ind, (data, targets) in enumerate(loader):
    # print(ind, data.shape, targets.shape)
    data = data.to('cuda', dtype = torch.float)
    targets = targets.float().unsqueeze(1).to(device=DEVICE)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        predictions = model(data)
        print(predictions.shape)
        loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
    print(ind, loss.item())
