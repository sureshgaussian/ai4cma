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

import logging

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
            preds = preds.cpu().detach().numpy().astype('uint8')
            if len(np.unique(preds)) > 2:
                print(f'{mask_tile_name[0]} has {np.unique(preds)} ')
                exit(0)

            #print("type of preds= ", type(preds), " shape of preds = ", preds.shape)
            assert(preds.shape[0] == len(mask_tile_name))
            
            for tile_idx in range(len(mask_tile_name)):
                out_file_path = os.path.join(tiled_output_dir, mask_tile_name[tile_idx])

                # cv2.imwrite(out_file_path, preds[0,0,:,:].astype('uint16'))
                #print(f'shape of cv2.imwrite is {preds[tile_idx,0,:,:].shape}')
                try:
                    cv2.imwrite(out_file_path, preds[tile_idx,0,:,:])
                except:
                    print("error in writing file: ", out_file_path)
                    exit()
    
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
    image = Image.open(output_file)
    assert(width == image.width)
    assert(height == image.height)
    image = np.array(image)
    if len(np.unique(image)) > 2:
        print(f'{input_file} has values >2 ')
        exit(0)

    
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

def infer_polys(in_tiles, input_file, label_fname, label_pattern_fname, label, 
                    legend_type, img_ht, img_wd, save_as_tiff, model, device, temp_inp_dir, temp_out_dir, results_dir, tile_size):
    inp_file_name = os.path.basename(input_file)
    input_descriptors = []
    assert len(in_tiles) > 0, "infer_polys called with no in_tiles"
    for in_tile in in_tiles:
        # print("in_tile = ", in_tile)
        splittext = in_tile.split("-")
        tile_nos  = [splittext[-2], splittext[-1]]
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

    return

def infer_points(input_file, points,output_file, save_as_tiff=True ):
    im=cv2.imread(input_file)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    xy_min, xy_max = points
    x_min, y_min = xy_min
    x_max, y_max = xy_max
    tx_min = min(x_min, x_max)
    tx_max = max(x_min, x_max)
    x_min = tx_min
    x_max = tx_max 

    ty_min = min(y_min, y_max)
    ty_max = max(y_min, y_max)
    y_min = ty_min
    y_max = ty_max 
    
    template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
    h, w = template.shape[0], template.shape[1]
    res = cv2.matchTemplate(im, template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.55
    loc = np.where( res >= threshold)
    
    # use the bounding boxes to create prediction binary raster
    pred_binary_raster=np.zeros((im.shape[0], im.shape[1]))
    for pt in zip(*loc[::-1]):
        #print('match found:')
        pred_binary_raster[int(pt[1]+float(h)/2), pt[0] + int(float(w)/2)]=1

    pred_binary_raster=pred_binary_raster.astype('uint16')
    cv2.imwrite(output_file, pred_binary_raster)
    if save_as_tiff:
        convert_mask_to_raster_tif(input_file, output_file)

def infer_lines(input_file, points,output_file, save_as_tiff=True ):


    im=cv2.imread(input_file)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    xy_min, xy_max = points
    x_min, y_min = xy_min
    x_max, y_max = xy_max

    tx_min = min(x_min, x_max)
    tx_max = max(x_min, x_max)
    x_min = tx_min
    x_max = tx_max 

    ty_min = min(y_min, y_max)
    ty_max = max(y_min, y_max)
    y_min = ty_min
    y_max = ty_max 
    
    template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
    h, w = template.shape[0], template.shape[1]
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    central_pixel=tuple(np.argwhere(edges==255)[0])
    sought = template[central_pixel].tolist()
    color_range=20
    lower = np.array([x - color_range for x in sought], dtype="uint8")
    upper = np.array([x + color_range for x in sought], dtype="uint8")
    
    # create a mask to only preserve current legend color in the basemap
    mask = cv2.inRange(im, lower, upper)
    detected = cv2.bitwise_and(im, im, mask=mask)
    
    # convert to grayscale 
    detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(detected_gray, 127, 255, cv2.THRESH_BINARY)[1]
    
    # convert the grayscale image to binary image
    pred_binary_raster = img_bw.astype(float) / 255


    pred_binary_raster=pred_binary_raster.astype('uint16')
    cv2.imwrite(output_file, pred_binary_raster)
    if save_as_tiff:
        convert_mask_to_raster_tif(input_file, output_file)        

def infer(input_dir, results_dir, temp_inp_dir, temp_out_dir, tile_size, save_as_tiff=True):
    # input_info_df = get_input_info(input_dir=input_dir)
    # check if the directories exist
    model, device = setup_inference()
    print(f'Setup of the model for inference complete')

    logging.basicConfig(filename="validation_run.log")


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

        print(f'Working on {json_file}')
        in_tiles = img2tiles.split_image_into_tiles(input_file, temp_inp_dir, tile_size)

        if in_tiles == None:
            continue
        
        with open(json_file) as jfile:
            label_info = json.load(jfile)

        img_ht = label_info["imageHeight"]
        img_wd = label_info["imageWidth"]

        # split the label_masks into tiles
        for shape in label_info["shapes"]:
            label = shape["label"]
            points = shape["points"]
            #if points ==
            print(f'Processing {input_file}, label: {label}')
            logging.info(f'Processing {input_file}, label: {label}')

            label_fname = os.path.splitext(json_file)[0]+"_"+label+".tif"
            label_fname = os.path.basename(label_fname)
            output_file = os.path.join(results_dir, label_fname)

            # for each label, get the label_pattern file
            label_pattern_fname = img2tiles.make_label_pattern(input_file, label, points, temp_inp_dir, tile_size)
            legend_type = label.split("_")[-1]
            if legend_type == "poly":
                infer_polys(in_tiles, input_file, label_fname, label_pattern_fname, label, 
                    legend_type, img_ht, img_wd, save_as_tiff, model, device, temp_inp_dir, temp_out_dir, results_dir, tile_size)
            else:
                if legend_type == "pt":
                    infer_points(input_file, points, output_file, save_as_tiff)
                else:
                    assert(legend_type == "line")
                    infer_lines(input_file, points, output_file, save_as_tiff)
            inp_file_name = os.path.basename(input_file)
            validation_info.append([os.path.basename(json_file), inp_file_name, img_ht, img_wd, legend_type, label_fname])

    df = pd.DataFrame(validation_info, columns=["json_file", "inp_file", "height", "width", "legend_type", "label_fname"])
    df.to_csv("inference_results.csv")
    return 

def prepare_for_submission():
    input_dir = INF_INP_DIR
    tinp_dir = INF_TEMP_TILED_INP_DIR
    tout_dir = INF_TEMP_TILED_OUT_DIR
    tile_size = TILE_SIZE
    results_dir = INF_RESULTS_DIR
    #create_legend_median_values(input_dir, output_json_file_name)
    infer(input_dir, results_dir, tinp_dir, tout_dir, tile_size)
    #convert_mask_to_raster_tif("../data/mini_validation/CO_Elkhorn.tif", "./temp/results/CO_Elkhorn_Qal_poly.tif")
    shutil.make_archive('gaussiansolutionsteam', format='zip', root_dir=results_dir)

    return 

def test_one_mask():
    model, device = setup_inference()
    csv_file = "predict.csv"
    tiled_input_dir = INF_TEMP_TILED_INP_DIR
    tiled_output_dir = INF_TEMP_TILED_OUT_DIR
    infer_one_mask(model, device, tiled_input_dir, csv_file, tiled_output_dir)

if __name__ == "__main__":
    #test_one_mask()
    prepare_for_submission()