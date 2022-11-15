import os
import sys
from utils_map import get_loader
import torch
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imshow_r
from utils_map import load_checkpoint

def inference(step):

    model = load_checkpoint('checkpoints/mapmodel_epoch_09.pth.tar')
    loader = get_loader(step)

    with torch.no_grad():
        for batch_ix, (x, y, img_names) in enumerate(loader):
            x = x.to("cuda", dtype=torch.float)
            y = y.to("cuda").unsqueeze(1)
            preds = model(x)['out']
            preds = (preds > 0.5).float()
            preds = preds.cpu().detach().numpy().astype('uint8')

            x = (x.cpu().detach().numpy()*255).astype('uint8')
            y = (y.cpu().detach().numpy()*255).astype('uint8')

            for img, mask, img_name, pred in zip(x, y, img_names, preds):

                img = np.moveaxis(img, 0, -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = np.squeeze(mask)

                pred_save_path = os.path.join('predictions_raw', step, img_name)
                cv2.imwrite(pred_save_path, pred[0])

                pred_contours = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(img, pred_contours, -1, (0, 0, 255), 2)

                target_contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(img, target_contours, -1, (0, 255, 0), 2)

                # imshow_r('img', img, True)

                viz_save_path = os.path.join('predictions_viz', step, img_name)
                cv2.imwrite(viz_save_path, img)


if __name__ == '__main__':
    step = 'validation'
    inference(step)