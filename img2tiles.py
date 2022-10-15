from math import nan, isnan
from turtle import width
import cv2
import rasterio
from PIL import Image
import os
import glob
import numpy as np
import json
import pandas as pd
Image.MAX_IMAGE_PIXELS = None


def split_image_into_tiles(input_file, output_dir, tile_size=256):
    """
    empty file concept introduced to reduce disk space usage
    """

    empty_file = os.path.join(output_dir, "empty_tile.tif")
    empty_file = os.path.abspath(empty_file)
    tx =0
    ty=0
    try:
        img = Image.open( input_file)
    except:
        print(f"error in opening {input_file}")
        print(f'yo yo yo')
        return None 
    xpatches = img.width // tile_size + 1
    ypatches = img.height // tile_size + 1
    tile_files = []
    filename = os.path.basename(input_file)
    for ty in range(0,ypatches):
        for tx in range(0,xpatches):
            bb = (tx*tile_size, ty*tile_size, (tx+1)*tile_size, (ty+1)*tile_size)
            crop_img = img.crop(bb)
            ext = "-"+str(tx)+"-"+str(ty)+".tif"
            tfile = os.path.join(output_dir, os.path.splitext(filename)[0]+ext)

            if np.sum(crop_img) != 0:
                crop_img.save(tfile)
            else:
                if not os.path.exists(empty_file):
                    crop_img.save(empty_file)
                
                tfile = os.path.abspath(tfile)

                if not os.path.exists(tfile):
                    os.symlink(src=empty_file, dst=tfile)

            tile_files.append(os.path.basename(tfile))
            
    return tile_files

def check_non_zero_tile(input_file):
    img_np = np.array(Image.open(input_file))
    if np.sum(img_np) != 0:
        return True
    else:
        return False



def get_non_zero_tiles(input_dir, filename):
    search_path = os.path.join(input_dir, os.path.splitext(filename)[0]+"_*")
    files = glob.glob(search_path)
    non_zero_tiles = []
    for file in files:
        non_zero = check_non_zero_tile(file)
        non_zero_tiles.append([file, non_zero])

    return non_zero_tiles


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))

def make_label_images(input_dir, input_file, label_file, output_dir, tile_size):
    # open the main file and get the dimensions
    img = Image.open( os.path.join(input_dir, input_file))
    width = img.width
    height = img.height
    with open(os.path.join(input_dir, label_file)) as jfile:
        label_info = json.load(jfile)

    label_patterns = []

    for shape in label_info["shapes"]:

        new_img = Image.new('RGB', (tile_size, tile_size))

        # read the label file
        label = shape["label"]
        label_bb = shape["points"]
        print(label)
        bb = bounding_box(label_bb)
        label_img = img.crop(bb)
        lx = label_img.width
        ly = label_img.height
        px = 0
        py = 0
        while px < tile_size:
            while py < tile_size:
                new_img.paste(label_img, (px,py))
                py = py + ly
            py = 0
            px = px+lx 
        ext = "-"+label+".tif"
        label_fname = os.path.join(output_dir, os.path.splitext(label_file)[0]+ext)
        new_img.save(label_fname)
        label_patterns.append(label_fname)

    return label_patterns
    
def make_label_pattern(input_file, label, label_bb, output_dir, tile_size):

    try:
        img = Image.open( input_file)
    except:
        print(f"error in opening {input_file}")
        print(f'yo yo yo')
        return None 
    width = img.width
    height = img.height
    new_img = Image.new('RGB', (tile_size, tile_size))

    # read the label file
    bb = bounding_box(label_bb)
    label_img = img.crop(bb)
    lx = label_img.width
    ly = label_img.height
    px = 0
    py = 0
    while px < tile_size:
        while py < tile_size:
            new_img.paste(label_img, (px,py))
            py = py + ly
        py = 0
        px = px+lx 
    ext = "_"+label+".tif"
    fname = os.path.basename(input_file)
    label_fname = os.path.join(output_dir, os.path.splitext(fname)[0]+ext)
    new_img.save(label_fname)
    return os.path.basename(label_fname)

def stitch_image_from_tiles(tile_size, base_filename, input_folder, output_filename, output_size=None, mask_flag=True):
    """
    Given a base file name, and a folder where to look for the tile image files, and the tile_size,
    this stitches the overall image, and save it.
    """
    print(f'base_filename = {base_filename}, input_folder = {input_folder}')
    #lets parse the list of files
    files = os.path.join(input_folder,base_filename)
    tile_files = glob.glob(files+"-*-*.tif")
    assert(len(tile_files) > 0)
    tfiles_info = []
    for tfile in tile_files:
        split_fname = os.path.basename(tfile)
        split_fname = os.path.splitext(split_fname)[0]
        tilex = int(split_fname.split("-")[-2])
        tiley = int(split_fname.split("-")[-1])
        # print(tfile, split_fname, tilex, tiley)
        tfiles_info.append( (tfile, split_fname, tilex, tiley))
    #print(type(tfiles_info))

    df = pd.DataFrame(tfiles_info, columns=['file_path', 'base_name', 'tile_x', 'tile_y'])
    max_tile_x = df['tile_x'].max()
    max_tile_y = df['tile_y'].max()
    print(f' max tile x, y are: {max_tile_x}, {max_tile_y}')
    if isnan(max_tile_x):
        print(df)

    # lets start filling in the new image
    if mask_flag:
        image = Image.new('L', (tile_size*max_tile_x, tile_size*max_tile_y))
    else:    
        image = Image.new('RGB', (tile_size*max_tile_x, tile_size*max_tile_y))

    print(f'Shape of empty image = {image.size}')

    for idx, row in df.iterrows():
        tile_img = Image.open(row['file_path']).convert('L')
        #tile_img = cv2.imread(row['file_path'], cv2.IMREAD_GRAYSCALE)
        #print(f'Shape of tile image = {tile_img.size}')

        tile_img = np.clip(np.array(tile_img), 0, 1)
        tile_img = Image.fromarray(tile_img)

        px = row['tile_x'] * tile_size
        py = row['tile_y'] * tile_size
        image.paste(tile_img, (px, py))

        
        if len(np.unique(tile_img)) > 2:
            print(f'{output_filename} has values >2 while stitching')
            exit(0)

    if output_size!= None:
        image = image.crop((0,0, output_size[0], output_size[1]))
    image.save(output_filename)

    image = np.array(image)
    if len(np.unique(image)) > 2:
        print(f'{output_filename} has values >2 after stitching')
        exit(0)


    return output_filename

def test_split_images_into_tiles():
    tile_size=256
    split_image_into_tiles("../data/training/", "AK_Bettles_ad_poly.tif", "./temp", tile_size)
    #make_label_images("../data/training/", "AK_Bettles.tif", "AK_Bettles.json", "./temp", tile_size)
    return 

def test_stitch_images():
    tile_size=256
    stitch_image_from_tiles(tile_size, "AK_Bettles_ad_poly", "./temp", "AK_Bettles_ad_poly_stitched.jpg", (512,512))
    return 

if __name__ == '__main__':
    
    test_split_images_into_tiles()
    test_stitch_images()
    