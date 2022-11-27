from config import *
from utils import draw_contours_big
from utils_show import imshow_r
import cv2
import pandas as pd
import imutils

if __name__ == '__main__':

    # Overlay predictions and ground truth
    step = 'validation'
    csv_path = os.path.join(TILED_INP_DIR, INFO_DIR, f"challenge_{step}_set.csv")
    df = pd.read_csv(csv_path)

    inp_dir = os.path.join(CHALLENGE_INP_DIR, 'training' if step == 'testing' else step)

    pred_dir = os.path.join(RESULTS_DIR, step)
    # pred_dir = os.path.join(POSTP_INMAP_DIR, step)
    # pred_dir = os.path.join(POSTP_OUTMAP_DIR, step)

    for ind, row in df.iterrows():

        save_path = os.path.join(RESULTS_CNTS_DIR, step, row['mask_fname'].replace('.tif', '.png'))
        if os.path.exists(save_path):
            continue

        if '_poly' not in row['mask_fname']:
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

        imshow_r(row['mask_fname'], img, stop=True, width=800)
        cv2.imwrite(save_path, imutils.resize(img, width=800))