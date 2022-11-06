import os
import torch

# common paths (DO NOT CHANGE)
ROOT_PATH = '/home/suresh/challenges/ai4cma'
CHALLENGE_INP_DIR = '/home/suresh/challenges/ai4cma/data'
MINI_CHALLENGE_INP_DIR = '/home/suresh/challenges/ai4cma/mini_data'
RESULTS_DIR = '/home/suresh/challenges/ai4cma/results'

TILED_INP_DIR = os.path.join(ROOT_PATH, 'tiled_inputs')
INFO_DIR = "info"
INPUTS_DIR = "inputs"
MASKS_DIR = "masks"
LEGENDS_DIR = "legends"
SPED_LEGENDS_DIR = 'sped_legends'
VALIDATION_DIR = "validation"

TRAIN_DATA_DIR = os.path.join(TILED_INP_DIR, 'challenge_training')
TRAIN_IMG_DIR = os.path.join(TRAIN_DATA_DIR, INPUTS_DIR)
TRAIN_LABEL_DIR = os.path.join(TRAIN_DATA_DIR, LEGENDS_DIR)
TRAIN_SPED_DIR = os.path.join(TRAIN_DATA_DIR, SPED_LEGENDS_DIR)
TRAIN_MASK_DIR = os.path.join(TRAIN_DATA_DIR, MASKS_DIR)
TRAIN_DESC = os.path.join(TRAIN_DATA_DIR, 'info/balanced_tiles.csv')

TEST_DATA_DIR = os.path.join(TILED_INP_DIR, 'challenge_testing')
TEST_IMG_DIR = os.path.join(TEST_DATA_DIR, INPUTS_DIR)
TEST_LABEL_DIR = os.path.join(TEST_DATA_DIR, LEGENDS_DIR)
TEST_MASK_DIR = os.path.join(TEST_DATA_DIR, MASKS_DIR)
TEST_DESC = os.path.join(TEST_DATA_DIR, 'info/balanced_tiles.csv')

WORKING_DIR = "/home/suresh/challenges/ai4cma/working_dir"

MODEL_NAME = "deeplabv3"
INF_MODEL_PATH = "submission_v1_model.pth.tar"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device = {DEVICE}')
BATCH_SIZE = 16
NUM_EPOCHS = 2
NUM_WORKERS = 4
TILE_SIZE = 256
IMAGE_HEIGHT = TILE_SIZE  # 1280 originally
IMAGE_WIDTH = TILE_SIZE  # 1918 originally
PIN_MEMORY = True
PERSISTANT_WORKERS = False
LOAD_MODEL = False
NUM_SAMPLES = None
TRAIN_TEST_SPLIT_RATIO = 0.8
EMPTY_TILES_RATIO=0.6
IN_CHANNELS = 6
USE_POST_PROCESSING = True
USE_AUGMENTATIONS = True

LOG_DIR = 'runs'

EXP_NAME = 'median_rgb_deeplabv3_mini'

EXP_NAME = f"{EXP_NAME}_{NUM_SAMPLES if NUM_SAMPLES else 'all'}"

CHEKPOINT_PATH = os.path.join('temp', f"my_checkpoint_{EXP_NAME}.pth.tar")
SAVED_IMAGE_PATH = os.path.join('temp', 'saved_images', EXP_NAME)
os.makedirs(SAVED_IMAGE_PATH, exist_ok=True)
