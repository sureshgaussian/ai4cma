SAVE_DEBUG_IMGS = False

import shutil
import os
if os.path.exists('debug_images'):
    shutil.rmtree('debug_images')
os.makedirs('debug_images', exist_ok=True)