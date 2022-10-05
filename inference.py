import rasterio 
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from zmq import device
import img2tiles 
import shutil 
import glob
from config import *
#from prepare import get_input_info
import pandas as pd 
import dataset 
from torch.utils.data import DataLoader

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn as nn
import torch
from utils import (
    load_checkpoint
)
import cv2
import json

def setup_inference():
    model = deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
    model.backbone.conv1 = nn.Conv2d(IN_CHANNELS, 64, 7, 2, 3, bias=False)
    if torch.cuda.is_available():
        model.cuda()
    load_checkpoint(torch.load(INF_MODEL_PATH), model)
    model.eval()
    device = DEVICE
    return model, device 

def infer_one_mask(model, device, tiled_input_dir, csv_file, tiled_output_dir):
    inp_csv = pd.read_csv(csv_file)
    cma_inference_dataset = dataset.CMAInferenceDataset(tiled_input_dir, 
                                tiled_input_dir, csv_file, None, USE_MEDIAN_COLOR)
    inference_loader = DataLoader(
        cma_inference_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    with torch.no_grad():
        for x, mask_tile_name in inference_loader:
            x = x.to(device, dtype=torch.float)
            preds = torch.sigmoid(model(x)['out'])
            preds = (preds > 0.5).float()
            #preds=preds.astype('uint16')
            preds = preds.cpu().detach().numpy()
            # print("type of preds= ", type(preds), " shape of preds = ", preds.shape)
            assert(preds.shape[0] == len(mask_tile_name))
            
            for tile_idx in range(len(mask_tile_name)):
                out_file_path = os.path.join(tiled_output_dir, mask_tile_name[tile_idx])

                # cv2.imwrite(out_file_path, preds[0,0,:,:].astype('uint16'))
                cv2.imwrite(out_file_path, preds[tile_idx,0,:,:])
    
    return 

def convert_mask_to_raster_tif(input_file, output_file):
    # convert the image to a binary raster .tif
    raster = rasterio.open(input_file)
    transform = raster.transform
    crs       = raster.crs 
    width     = raster.width 
    height    = raster.height 
   
    raster.close()
    # image = Image.open(output_file)
    image = Image.open(output_file).convert("L")
    assert(width == image.width)
    assert(height == image.height)
    image = np.array(image)
    
    with rasterio.open(output_file, 'w', 
                        driver    = 'GTIFF', 
                        transform = transform, 
                        dtype     = rasterio.uint8, 
                        count     = 1, 
                        compress  = 'lzw', 
                        crs       = crs, 
                        width     = width, 
                        height    = height) as dst:
        
        dst.write(image, indexes=1)
        dst.close()    

def infer(input_dir, results_dir, temp_inp_dir, temp_out_dir, tile_size, save_as_tiff=True):
    # input_info_df = get_input_info(input_dir=input_dir)
    # check if the directories exist
    model, device = setup_inference()

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    validation_info = []
    for json_file in json_files:
        input_descriptors = []
        input_file = json_file.replace(".json",".tif")
        if not os.path.isdir(temp_inp_dir):
            os.mkdir(temp_inp_dir)
        else:
            #delete the directory to delete all the files under it
            # recreate the folder for new inputs
            shutil.rmtree(temp_inp_dir)
            os.mkdir(temp_inp_dir)

        in_tiles = img2tiles.split_image_into_tiles(input_file, temp_inp_dir, tile_size)
        
        with open(json_file) as jfile:
            label_info = json.load(jfile)

        img_ht = label_info["imageHeight"]
        img_wd = label_info["imageWidth"]

        # split the label_masks into tiles
        for shape in label_info["shapes"]:
            label = shape["label"]
            points = shape["points"]

            label_fname = os.path.splitext(json_file)[0]+"_"+label+".tif"
            label_fname = os.path.basename(label_fname)

            # for each label, get the label_pattern file
            label_pattern_fname = img2tiles.make_label_pattern(input_file, label, points, temp_inp_dir, tile_size)
            legend_type = label.split("_")[-1]
            if legend_type != "poly":
                continue

            inp_file_name = os.path.basename(input_file)
            input_descriptors = []
            for in_tile in in_tiles:
                # print("in_tile = ", in_tile)
                splittext = in_tile.split("-")
                tile_nos  = [splittext[1], splittext[2]]
                mask_ext = "-"+str(tile_nos[0])+"-"+str(tile_nos[1])
                mask_tile_name = os.path.splitext(label_fname)[0]+mask_ext
                # print("mask_tile = ", mask_name)
                # break
                input_descriptors.append([inp_file_name, in_tile, label_pattern_fname, mask_tile_name, label, legend_type, img_ht, img_wd])
            
            if not os.path.isdir(temp_out_dir):
                os.mkdir(temp_out_dir)
            
            # input_descriptors ready to be shipped to the model
            inp_df = pd.DataFrame(input_descriptors, columns = ["inp_file_name", "in_tile", "label_pattern_fname", "mask_tile_name", "legend", "legend_type", "height", "width"])
            inp_df.to_csv("predict.csv", index=False)

            # ship to inference on one image
            infer_one_mask(model, device, temp_inp_dir, "predict.csv", temp_out_dir)

            # get the original image from tiles
            label_fname = os.path.splitext(label_pattern_fname)[0]
            mask_out_fname = os.path.join(results_dir, label_fname+".tif")
            img2tiles.stitch_image_from_tiles(tile_size, label_fname, temp_out_dir, mask_out_fname, (img_wd, img_ht))

            if save_as_tiff:
                convert_mask_to_raster_tif(input_file, os.path.join(results_dir, mask_out_fname))


            #remove the output directory
            assert(os.path.isdir(temp_out_dir))
            shutil.rmtree(temp_out_dir)
            os.mkdir(temp_out_dir)

            validation_info.append([os.path.basename(json_file), inp_file_name, img_ht, img_wd, legend_type, label_fname])

    df = pd.DataFrame(validation_info, columns=["json_file", "inp_file", "height", "width", "legend_type", "label_fname"])
    df.to_csv("inference_results.csv")
    return 


if __name__ == "__main__":
    input_dir = INF_INP_DIR
    tinp_dir = INF_TEMP_TILED_INP_DIR
    tout_dir = INF_TEMP_TILED_OUT_DIR
    tile_size = TILE_SIZE
    results_dir = INF_RESULTS_DIR
    infer(input_dir, results_dir, tinp_dir, tout_dir, tile_size)
    #convert_mask_to_raster_tif("../data/mini_validation/CO_Elkhorn.tif", "./temp/results/CO_Elkhorn_Qal_poly.tif")
    shutil.make_archive('gaussiansolutionsteam', format='zip', root_dir=results_dir)