from turtle import width
from PIL import Image
import os
import glob
import numpy as np
import json
import pandas as pd


def split_image_into_tiles(input_dir, filename, output_dir, tile_size=256):
    tx =0
    ty=0
    img = Image.open( os.path.join(input_dir, filename))
    xpatches = img.width // tile_size + 1
    ypatches = img.height // tile_size + 1
    tile_files = []
    for ty in range(0,ypatches):
        for tx in range(0,xpatches):
            bb = (tx*tile_size, ty*tile_size, (tx+1)*tile_size, (ty+1)*tile_size)
            crop_img = img.crop(bb)
            ext = "-"+str(tx)+"-"+str(ty)+".jpg"
            tfile = os.path.join(output_dir, os.path.splitext(filename)[0]+ext)
            crop_img.save(tfile)
            tile_files.append(tfile)
            
    return tile_files

def get_non_zero_tiles(input_dir, filename):
    search_path = os.path.join(input_dir, os.path.splitext(filename)[0]+"_*")
    files = glob.glob(search_path)
    non_zero_tiles = []
    for file in files:
        img_np = np.array(Image.open(file))
        if np.sum(img_np) != 0:
            non_zero_tiles.append([file, np.sum(img_np)])

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
        ext = "-"+label+"-pattern.jpg"
        label_fname = os.path.join(output_dir, os.path.splitext(label_file)[0]+ext)
        new_img.save(label_fname)
        label_patterns.append(label_fname)

    return label_patterns
    
def make_label_pattern(input_dir, input_file, label, label_bb, output_dir, tile_size):
    img = Image.open( os.path.join(input_dir, input_file))
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
    ext = "-"+label+"-pattern.jpg"
    label_fname = os.path.join(output_dir, os.path.splitext(input_file)[0]+ext)
    new_img.save(label_fname)
    return label_fname

def stitch_image_from_tiles(tile_size, base_filename, input_folder, output_filename, output_size=None):
    """
    Given a base file name, and a folder where to look for the tile image files, and the tile_size,
    this stitches the overall image, and save it.
    """
    #lets parse the list of files
    files = os.path.join(input_folder,base_filename)
    tile_files = glob.glob(files+"-*-*.jpg")
    tfiles_info = []
    for tfile in tile_files:
        split_fname = os.path.basename(tfile)
        split_fname = os.path.splitext(split_fname)[0]
        tilex = int(split_fname.split("-")[1])
        tiley = int(split_fname.split("-")[2])
        print(tfile, split_fname, tilex, tiley)
        tfiles_info.append( (tfile, split_fname, tilex, tiley))
    print(type(tfiles_info))

    df = pd.DataFrame(tfiles_info, columns=['file_path', 'base_name', 'tile_x', 'tile_y'])
    max_tile_x = df['tile_x'].max()
    max_tile_y = df['tile_y'].max()
    print(f'{max_tile_x}, {max_tile_y}')

    # lets start filling in the new image
    image = Image.new('RGB', (tile_size*max_tile_x, tile_size*max_tile_y))
    for idx, row in df.iterrows():
        tile_img = Image.open(row['file_path'])
        
        px = row['tile_x'] * tile_size
        py = row['tile_y'] * tile_size
        image.paste(tile_img, (px, py))

    if output_size!= None:
        image = image.crop((0,0, output_size[0], output_size[1]))
    image.save(output_filename)

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
    