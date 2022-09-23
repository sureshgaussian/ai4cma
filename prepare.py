from cgitb import reset
import pandas as pd
import csv 
import os
import glob
from posixpath import splitext
import img2tiles
import json

def prepare_inputs(input_dir, output_dir, tile_size=256, tiled_input_dir="tiled_inputs",
                    tiled_label_dir = "tiled_labels", tiled_mask_dir = "tiled_masks"):
    jfiles_path = os.path.join(input_dir, "*.json")
    print(jfiles_path)
    json_files = glob.glob(jfiles_path)
    inputs_descriptor = []
    tinput_dir = os.path.join(output_dir, tiled_input_dir)
    if not os.path.isdir(tinput_dir):
        print(f'Creating the directory: {tinput_dir}')
        os.mkdir(tinput_dir)
    
    label_mask_dir = os.path.join(output_dir, tiled_mask_dir)

    if not os.path.isdir(label_mask_dir):
        print(f'Creating the directory: {label_mask_dir}')
        os.mkdir(label_mask_dir)

    legend_pattern_dir = os.path.join(output_dir, tiled_label_dir)

    if not os.path.isdir(legend_pattern_dir):
        print(f'Creating the directory: {legend_pattern_dir}')
        os.mkdir(legend_pattern_dir)




    for json_file in json_files:

        # find corresponding input image
        print(json_file)
        input_file = json_file.replace(".json",".tif")
        
        in_tiles = img2tiles.split_image_into_tiles(input_file, tinput_dir, tile_size)

        # prepare the label_pattern_tile for all labels
        j_fname = os.path.split(json_file)[-1]
        #label_patterns = img2tiles.make_label_images(input_dir, in_fname, j_fname, output_dir, tile_size)


        with open(json_file) as jfile:
             label_info = json.load(jfile)

        img_ht = label_info["imageHeight"]
        img_wd = label_info["imageWidth"]

        # split the label_masks into tiles
        for shape in label_info["shapes"]:
            label = shape["label"]
            points = shape["points"]
            

            #get the label.tif file
            label_fname = os.path.splitext(j_fname)[0]+"_"+label+".tif"
            label_input_file = os.path.join(input_dir, label_fname )
            label_mask_tiles = img2tiles.split_image_into_tiles(label_input_file, label_mask_dir, tile_size)

            # for each label, get the label_pattern file
            label_pattern_fname = img2tiles.make_label_pattern(input_file, label, points, legend_pattern_dir, tile_size)

            inp_file_name = os.path.basename(input_file)

            assert(len(label_mask_tiles) == len(in_tiles))
            for idx, tile in enumerate(label_mask_tiles):
                empty_tile = img2tiles.check_non_zero_tile(os.path.join(label_mask_dir, label_mask_tiles[idx]))
                inputs_descriptor.append([inp_file_name, img_ht, img_wd, in_tiles[idx], label_pattern_fname, label_mask_tiles[idx], empty_tile, tile_size])

            print("Current length of inputs: ", len(inputs_descriptor))
            break
    
    return inputs_descriptor

def prepare_balanced_inputs(input_csv_file, output_train_csv_file, output_test_csv_file, ratio=0.6, test_split=0.2):
    """
    Given input csv file, select files for training
    """
    df = pd.read_csv(input_csv_file)
    avg_true = df['empty_tile'].mean()
    true_df = df[df['empty_tile']==True]
    true_train_df = true_df.sample(frac=1-test_split).reset_index(drop=True)
    true_test_df=true_df[~true_df.isin(true_train_df)].dropna(how = 'all')

    if avg_true < ratio:
        false_df = df[df['empty_tile']==False]
        fratio = avg_true/ratio 
        false_df_shuffle = false_df.sample(frac=fratio).reset_index(drop=True)

        false_train_df = false_df_shuffle.sample(frac=1-test_split).reset_index(drop=True)
        false_test_df=false_df_shuffle[~false_df_shuffle.isin(false_train_df)].dropna(how = 'all')

        input_train_df = pd.concat([true_train_df, false_train_df])
        input_train_df_shuffled = input_train_df.sample(frac=1).reset_index(drop=True)

        input_test_df = pd.concat([true_test_df, false_test_df])
        input_test_df_shuffled = input_train_df.sample(frac=1).reset_index(drop=True)
    else:
        input_train_df_shuffled = df.sample(frac=1-test_split).reset_index(drop=True)
        input_test_df_shuffled = df[~df.isin(input_train_df)].dropna(how= 'all')
    
    input_train_df_shuffled.to_csv(output_train_csv_file)
    input_test_df_shuffled.to_csv(output_test_csv_file)

    print(f'Avg_True  {input_train_df_shuffled["empty_tile"].mean()}')
    return 

def test_prepare_inputs():
    tile_size = 256
    input_descriptors = prepare_inputs("./inp", "./temp", tile_size)
    print(type(input_descriptors))
    
    df = pd.DataFrame(input_descriptors, columns = ["orig_file", "orig_ht", "orig_wd", "tile_inp", "tile_legend", "tile_mask", "empty_tile", "tile_size"])
    df.to_csv("input.csv", index=False)



if __name__ == '__main__':
    
    prepare_balanced_inputs("input.csv", "train.csv", "test.csv")
