from math import nan, isnan
import cv2
import rasterio
from PIL import Image
import os
import glob
import numpy as np
import json
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
from preprocess import preprocess_legend_coordinates
from utils_show import imshow_r


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

def scale_pack_legend(input_tif_file, legend_bb, output_sp_legened_file, tile_size=TILE_SIZE):
    try:
        img = Image.open(input_tif_file)
    except:
        print(f"error in opening {input_tif_file}")
        print(f'yo yo yo')
        return None 
    width = img.width
    height = img.height
    new_img = Image.new('RGB', (tile_size, tile_size))

    # read the label file
    legend_bb = preprocess_legend_coordinates(legend_bb)
    bb = bounding_box(legend_bb)
    label_img = img.crop(bb)
    imshow_r('label', label_img)
    # tile_siz/4 rescaled, and rotated image
    assert(tile_size % 16 == 0)
    scaled_by_4 = label_img.resize((tile_size//2, tile_size//2))
    scaled_by_4_90d = scaled_by_4.rotate(90)
    scaled_by_4_270d = scaled_by_4.rotate(270)
    scaled_by_8 = label_img.resize((tile_size//4, tile_size//4))
    scaled_by_8_90d = scaled_by_8.rotate(90)
    scaled_by_8_45d = scaled_by_8.rotate(45, fillcolor=(255,255,255))
    scaled_by_8_315d = scaled_by_8.rotate(315, fillcolor=(255,255,255))
    new_img.paste(scaled_by_4, (0,0))
    new_img.paste(scaled_by_4_90d, (tile_size//2,0))
    new_img.paste(scaled_by_4_270d, (0,tile_size//2))

    new_img.paste(scaled_by_8, (tile_size//2, tile_size//2))
    new_img.paste(scaled_by_8_90d, (tile_size//2+tile_size//4, tile_size//2))
    new_img.paste(scaled_by_8_45d, (tile_size//2+tile_size//4, tile_size//2+tile_size//4))
    new_img.paste(scaled_by_8_315d, (tile_size//2, tile_size//2+tile_size//4))

    scaled_by_16 = label_img.resize((tile_size//8, tile_size//8))
    scaled_by_16_90d = scaled_by_16.rotate(90)
    scaled_by_16_45d = scaled_by_16.rotate(45)
    scaled_by_16_225d = scaled_by_16.rotate(225)
    scaled_by_16_22d = scaled_by_16.rotate(22)
    scaled_by_16_247d = scaled_by_16.rotate(247)
    scaled_by_16_67d = scaled_by_16.rotate(67)
    scaled_by_16_203d = scaled_by_16.rotate(203)

    new_img.paste(scaled_by_16, (tile_size//2, tile_size//2))
    new_img.paste(scaled_by_16_90d, (tile_size//2, tile_size//8+tile_size//2))
    new_img.paste(scaled_by_16_45d, (tile_size//2, tile_size//4+tile_size//2))
    new_img.paste(scaled_by_16_225d, (tile_size//2, tile_size//2+3*tile_size//8))
    new_img.paste(scaled_by_16_67d, (tile_size//2+tile_size//8, tile_size//2))
    new_img.paste(scaled_by_16_22d, (tile_size//2+tile_size//8, tile_size//8+tile_size//2))
    new_img.paste(scaled_by_16_247d, (tile_size//2+tile_size//8, tile_size//4+tile_size//2))
    new_img.paste(scaled_by_16_203d, (tile_size//2+tile_size//8, tile_size//2+3*tile_size//8))
    temp_img = new_img.crop((tile_size//2, tile_size//2, tile_size//2+tile_size//4, tile_size//2+tile_size//4))
    temp_img = temp_img.rotate(15)
    new_img.paste(temp_img, (tile_size//2+tile_size//4, tile_size//2))

    new_img.save(output_sp_legened_file)
    # return os.path.basename(output_sp_legened_file)
    return

def main(args):
    inp_csv = args.info_csv
    tile_size = args.tile_size
    output_dir = args.output_dir
    input_dir = args.input_dir
    csv_df = pd.read_csv(inp_csv)
    total_files = len(csv_df)
    for idx, row in csv_df.iterrows():
        print(row)
        if 'poly' in row['label']:
            continue
        if idx % 10 == 0:
            print(f'finished processing {idx} out of {total_files}...')
        # output_legend_file = os.path.join(output_dir, row['mask_fname']).replace('.tif', '.png')
        output_legend_file = os.path.join(output_dir, row['mask_fname'])
        input_tif_file = os.path.join(input_dir, row['inp_fname'])
        bb = ast.literal_eval(row['points'])
        scale_pack_legend(input_tif_file, bb, output_legend_file, tile_size=tile_size)
        # break

def run_in_map():
    in_df = pd.read_csv("../tiled_inputs/challenge_testing/info/balanced_tiles.csv")
    # in_df = in_df.sort_values(by='orig_file')
    out_df = check_in_map(in_df, "../downscaled_data/masks_upscaled/")
    # out_df.sort_index(axis=0)
    out_df.to_csv("../tiled_inputs/challenge_testing/info/balanced_tiles_in_map.csv", index=False)
    print(np.sum(out_df.in_map))


if __name__ == '__main__':
    
    test_split_images_into_tiles()
    test_stitch_images()
    