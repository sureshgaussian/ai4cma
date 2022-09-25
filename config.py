import os
import torch

CODE_DIR = '/home/suresh/challenges/ai4cma/code'
IMG_DIR = os.path.join(CODE_DIR, 'temp/tiled_inputs')
LABEL_DIR = os.path.join(CODE_DIR, 'temp/tiled_labels')
MASK_DIR = os.path.join(CODE_DIR, 'temp/tiled_masks')
TRAIN_DESC = os.path.join(CODE_DIR, "train.csv")
VAL_DESC = os.path.join(CODE_DIR, "test.csv")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device = {DEVICE}')
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
TILE_SIZE = 256
IMAGE_HEIGHT = TILE_SIZE  # 1280 originally
IMAGE_WIDTH = TILE_SIZE  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
NUM_SAMPLES = None

#Ravi's experiments
USE_MEDIAN_COLOR = True
if USE_MEDIAN_COLOR:
    EXP_NAME = 'median_rgb_deeplabv3'
    IN_CHANNELS = 6
else:
    EXP_NAME = 'rgb'
    IN_CHANNELS = 6

EXP_NAME = f"{EXP_NAME}_{NUM_SAMPLES if NUM_SAMPLES else 'all'}"

CHEKPOINT_PATH = os.path.join('temp', f"my_checkpoint_{EXP_NAME}.pth.tar")
SAVED_IMAGE_PATH = os.path.join('temp', 'saved_images', EXP_NAME)
os.makedirs(SAVED_IMAGE_PATH, exist_ok=True)
