import argparse
from ast import operator 
from cgitb import reset
import pandas as pd
import csv 
import os
import glob
import numpy as np
from posixpath import splitext
from config import (
    CHALLENGE_INP_DIR, 
    MINI_CHALLENGE_INP_DIR, 
    TILED_INP_DIR, INFO_DIR, TILE_SIZE,
    INPUTS_DIR, MASKS_DIR, LEGENDS_DIR,
    TRAIN_TEST_SPLIT_RATIO,
    EMPTY_TILES_RATIO
)
import img2tiles
import json


def make_output_dirs(input_dir, output_dir, tiled_input_dir="tiled_inputs",
                     tiled_label_dir = "tiled_labels", tiled_mask_dir = "tiled_masks"):
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

    return tinput_dir, label_mask_dir, legend_pattern_dir

def prepare_inputs_from_json_files(json_files, input_dir, output_dir, tile_size=256, tiled_input_dir="tiled_inputs",
                    tiled_label_dir = "tiled_labels", tiled_mask_dir = "tiled_masks"):
    inputs_descriptor = []
    tinput_dir, label_mask_dir, legend_pattern_dir = make_output_dirs(input_dir, output_dir, tiled_input_dir, tiled_label_dir , tiled_mask_dir )


    for json_file in json_files:

        # find corresponding input image
        print(json_file)
        input_file = json_file.replace(".json",".tif")
        
        in_tiles = img2tiles.split_image_into_tiles(input_file, tinput_dir, tile_size)
        if in_tiles == None:
            print(f'skipping {json_file} as there is no file')
            continue

        # prepare the label_pattern_tile for all labels
        # j_fname = os.path.basename(json_file)
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
            label_fname = os.path.splitext(json_file)[0]+"_"+label+".tif"
            # label_input_file = os.path.join(input_dir, label_fname )
            label_mask_tiles = img2tiles.split_image_into_tiles(label_fname, label_mask_dir, tile_size)

            # for each label, get the label_pattern file
            label_pattern_fname = img2tiles.make_label_pattern(input_file, label, points, legend_pattern_dir, tile_size)

            inp_file_name = os.path.basename(input_file)

            assert(len(label_mask_tiles) == len(in_tiles))
            for idx, tile in enumerate(label_mask_tiles):
                empty_tile = img2tiles.check_non_zero_tile(os.path.join(label_mask_dir, label_mask_tiles[idx]))
                inputs_descriptor.append([inp_file_name, img_ht, img_wd, in_tiles[idx], label_pattern_fname, label_mask_tiles[idx], empty_tile, tile_size])

            print("Current length of inputs: ", len(inputs_descriptor))
            
    
    return inputs_descriptor

def prepare_inputs_from_csv(csv_file, input_dir, output_dir, tile_size=256, tiled_input_dir="tiled_inputs",
                    tiled_label_dir = "tiled_labels", tiled_mask_dir = "tiled_masks"):

    df = pd.read_csv(csv_file)
    input_files = df.inp_fname.unique()
    json_files = [ os.path.join(input_dir, x.replace(".tif", ".json")) for x in input_files]
    # json_files = df.filepath_name.to_list()

    #json_files = json_files[0:2]

    inputs_descriptor = prepare_inputs_from_json_files(json_files, input_dir, output_dir, tile_size, tiled_input_dir,
                    tiled_label_dir , tiled_mask_dir )
    return inputs_descriptor
    

def prepare_inputs(input_dir, output_dir, tile_size=256, tiled_input_dir="tiled_inputs",
                    tiled_label_dir = "tiled_labels", tiled_mask_dir = "tiled_masks"):
    jfiles_path = os.path.join(input_dir, "*.json")
    print(jfiles_path)
    json_files = glob.glob(jfiles_path)
    inputs_descriptor = prepare_inputs_from_json_files(json_files, input_dir, output_dir, tile_size, tiled_input_dir,
                    tiled_label_dir , tiled_mask_dir )
    return inputs_descriptor


def prepare_balanced_inputs(input_csv_file, output_train_csv_file, output_test_csv_file, ratio=0.6, test_split=0.2):
    """
    Given input csv file, select files for training
    """
    df = pd.read_csv(input_csv_file)
    avg_true = df['empty_tile'].mean()
    print(f'avg_true of the dataset = {avg_true}')
    true_df = df[df['empty_tile']==True]
    #true_train_df = true_df.sample(frac=1-test_split).reset_index(drop=True)
    true_train_df = true_df.sample(frac=1-test_split)
    true_test_df=true_df[~true_df.isin(true_train_df)].dropna(how = 'all')
    #true_test_df = pd.concat([true_df, true_train_df, true_train_df]).drop_duplicates(keep=False)

    print(f'length of true = {len(true_df)}')
    print(f'length of true_train = {len(true_train_df)}')
    print(f'length of true_test = {len(true_test_df)}')

    if avg_true < ratio:
        false_df = df[df['empty_tile']==False]
        fratio = avg_true/ratio 
        print(f'{fratio}')
        false_df_shuffle = false_df.sample(frac=fratio)


        false_train_df = false_df_shuffle.sample(frac=1-test_split)
        false_test_df=false_df_shuffle[~false_df_shuffle.isin(false_train_df)].dropna(how = 'all')

        input_train_df = pd.concat([true_train_df, false_train_df])
        input_train_df_shuffled = input_train_df.sample(frac=1)

        input_test_df = pd.concat([true_test_df, false_test_df])
        input_test_df_shuffled = input_test_df.sample(frac=1)
    else:
        input_train_df_shuffled = df.sample(frac=1-test_split)
        input_test_df_shuffled = df[~df.isin(input_train_df_shuffled)].dropna(how= 'all')
    
    input_train_df_shuffled.to_csv(output_train_csv_file)
    input_test_df_shuffled.to_csv(output_test_csv_file)

    print(f'Avg_True  {input_train_df_shuffled["empty_tile"].mean()}')
    print(f'Avg_True test {input_test_df_shuffled["empty_tile"].mean()}')
    print(f'train len: {len(input_train_df_shuffled)}, test len: {len(input_test_df_shuffled)}')
    return 

def prepare_balanced_empty_tiles(input_csv, output_csv, ratio=0.6):
    df = pd.read_csv(input_csv)
    # balance empty tiles vs non-empty
    empty_ratio = len(df[df.empty_tile == False])/len(df)
    print(empty_ratio)

    if empty_ratio > ratio:
        true_df = df[df.empty_tile==True]
        # ratio = fl/(fl+tl)
        # ratio(fl+tl) = fl
        # (1-ratio)*f1= ratio*tl
        # fl = ratio*tl(1-ratio)
        false_len = int(ratio*len(true_df)/(1-ratio))
        false_df = df[df.empty_tile == False]
        false_df = false_df.sample(n=false_len, random_state=42)
        input_df = pd.concat([false_df, true_df])
        input_df = input_df.sample(frac=1, random_state=43)
    else:
        input_df = df 
    
    print(f'Input df len = {len(df)}, output = {len(input_df)} ')
    
    input_df.to_csv(output_csv, index=False)
    return 

def get_input_info(input_dir):
    jfiles_path = os.path.join(input_dir, "*.json")
    #print(jfiles_path)
    json_files = glob.glob(jfiles_path)
    inputs_descriptor = []
    for json_file in json_files:

        input_file = json_file.replace(".json",".tif")
        
        # prepare the label_pattern_tile for all labels
        j_fname = os.path.basename(json_file)
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
            legend_type = label.split("_")[-1]
            inp_file_name = os.path.basename(input_file)
            inputs_descriptor.append([inp_file_name, label_fname, label, legend_type, img_wd, img_ht, points])
    
    df = pd.DataFrame(inputs_descriptor, columns=["inp_fname", "mask_fname", "label", "legend_type", "width", "height", "points"])      
    return df

def test_get_input_info():
    info = get_input_info(os.path.join(DATA_DIR, 'training'))
    info.to_csv(os.path.join(DATA_DIR, "training_input_info.csv"), index=False)

def test_prepare_inputs(training_csv_file="training_tiled_inputs.csv"):
    tile_size = 256
    #input_descriptors = prepare_inputs("../data/training", "../data/training_inp", tile_size)
    # input_descriptors =prepare_inputs_from_csv(os.path.join(EDA_DIR, 'train_split.csv'), os.path.join(DATA_DIR, 'training') , os.path.join(CODE_DIR, 'temp'), tile_size)
    input_descriptors =prepare_inputs_from_csv(os.path.join(EDA_DIR, 'test_split.csv'), os.path.join(DATA_DIR, 'training') , os.path.join(CODE_DIR, 'temp'), tile_size)
    #print(type(input_descriptors))
    
    df = pd.DataFrame(input_descriptors, columns = ["orig_file", "orig_ht", "orig_wd", "tile_inp", "tile_legend", "tile_mask", "empty_tile", "tile_size"])
    df.to_csv(training_csv_file, index=False)

def old_main():
    # training_csv_file="training_tiled_inputs.csv"
    training_csv_file= os.path.join(CODE_DIR, "test_tiled_inputs.csv")
    test_prepare_inputs(training_csv_file)
    #prepare_balanced_inputs(training_csv_file, "train.csv", "test.csv")
    #test_get_input_info()
    output_csv = os.path.join(CODE_DIR, "balanced_tiled_test.csv")
    prepare_balanced_empty_tiles(training_csv_file, output_csv)


def split_train_test(input_dir):
    info_dir = os.path.join(TILED_INP_DIR, INFO_DIR)
    print(f'running get_input_info on {input_dir}')
    training_df = get_input_info(input_dir)
    in_files = training_df.inp_fname.unique()
    ratio = TRAIN_TEST_SPLIT_RATIO
    training_files = np.random.choice(in_files, size= int(ratio*len(in_files)), replace=False )
    test_files = set(in_files) - set(training_files)

    print(f'length of input files: {len(in_files)}, training len: {len(training_files)} , testing len: {len(test_files)}')
    
    train_df = training_df[training_df.inp_fname.isin(training_files)]
    train_split_csv = os.path.join(info_dir, args.dataset+"_training_files.csv")
    train_df.to_csv(train_split_csv, index=False)
    #df['filename'] = df['filepath_name']
    test_df = training_df[training_df.inp_fname.isin(test_files)]
    test_split_csv = os.path.join(info_dir, args.dataset+"_testing_files.csv") 
    test_df.to_csv( test_split_csv, index=False)
    print(f'train_test_split operation on : {input_dir}, info written into: {info_dir} , in {train_split_csv} & {test_split_csv} files')

def tilize_inputs(input_dir, stage, dataset, tile_size):
    print(f"preparing inputs for {stage} using dataset: {dataset}")
    output_dir = os.path.join(TILED_INP_DIR, dataset+ "_"+stage)
    if not os.path.isdir(output_dir):
        print(f'Creating the directory: {output_dir}')
        os.mkdir(output_dir)
    print(f'input directory = {input_dir}, output dir = {output_dir}')

    csv_file = dataset+"_"+stage+"_files.csv"

    tiled_input_dir = INPUTS_DIR
    tiled_masks_dir = MASKS_DIR
    tiled_legends_dir = LEGENDS_DIR
    input_csv_file = os.path.join(TILED_INP_DIR,INFO_DIR)
    input_csv_file = os.path.join(input_csv_file, csv_file)
    print(f" taking the inputs from: {input_csv_file}")
    input_descriptors = prepare_inputs_from_csv(input_csv_file, input_dir, output_dir, tile_size, 
                            tiled_input_dir, tiled_legends_dir, tiled_masks_dir)
    
    # TODO: push this into prepare function?
    csv_file_dir = os.path.join(output_dir, INFO_DIR)
    if not os.path.isdir(csv_file_dir):
        print(f'Creating the directory: {csv_file_dir}')
        os.mkdir(csv_file_dir)

    
    csv_file = os.path.join(csv_file_dir, "all_tiles.csv")
    df = pd.DataFrame(input_descriptors, columns = ["orig_file", "orig_ht", "orig_wd", "tile_inp", "tile_legend", "tile_mask", "empty_tile", "tile_size"])

    df.to_csv(csv_file, index=False)
    balanced_csv_files = os.path.join(csv_file_dir, "balanced_tiles.csv")
    prepare_balanced_empty_tiles(csv_file, balanced_csv_files, ratio=EMPTY_TILES_RATIO)



    
def process_args(args):
    stages_dir = {
        'training': "training",
        'testing' : "training", 
        'validation' : "validation"}
    switch_dataset = {
        'mini': MINI_CHALLENGE_INP_DIR,
        'challenge': CHALLENGE_INP_DIR,
    }


    assert( str(args.tile_size).isnumeric() )

    if args.stage not in stages_dir.keys():
        print('Undefined stage: ', args.stage, ' knowns stages: ', stages_dir.keys())
        return 
    if args.dataset in switch_dataset.keys():
        base_input_dir = switch_dataset[args.dataset] 
    else:
        print("Unsupported dataset: ", args.dataset, " chose from: ", switch_dataset.keys() )
        return

    input_dir = os.path.join(base_input_dir, args.stage)
    
    if args.operation == 'tilize_inputs':
        if args.stage == 'validation':
            print(f'Operation {args.operation} not valid for stage: {args.stage}')
            return
        input_dir = os.path.join(base_input_dir, "training")
        tilize_inputs(input_dir, args.stage, args.dataset, args.tile_size)

    elif args.operation == 'get_input_info':
        info_dir = os.path.join(TILED_INP_DIR, INFO_DIR)
        output_csv = os.path.join(info_dir, args.dataset+"_"+args.stage+"_set.csv")
        print('getting info about the inputs in : ', input_dir, " into: ", output_csv)
        df = get_input_info(input_dir)
        df.to_csv(output_csv, index=False)

    elif args.operation == 'train_test_split':
        split_train_test(input_dir)
       
    elif args.operation == 'prepare_inputs':
        if args.stage != 'training':
            print(f'Operation {args.operation} not valid for stage: {args.stage}')
            return
        print(f'Running split_train_test on {input_dir}')
        split_train_test(input_dir)
        print(f'Tilizing the training split')
        tilize_inputs(input_dir, "training", args.dataset, args.tile_size)
        print(f'Tilizing the testing split')
        tilize_inputs(input_dir, "testing", args.dataset, args.tile_size)


        
    else:
        print('unsupported operations')
        return 

def prepare_file_train_test_split(input_desc_csv, ratio=0.8):
    full_df = pd.read_csv(input_desc_csv)


if __name__ == '__main__':
# prepare -s {training, testing, mini_training, validation}
    parser = argparse.ArgumentParser(description='Prepare inputs parser')
    parser.add_argument('-s', '--stage', default='training', help='which stage? [training, testing, validation]')
    parser.add_argument('-d', '--dataset', default='mini', help='which dataset [ mini, challenge]')
    parser.add_argument('-o', '--operation', default='train_test_split', help='operations:[prepare_inputs, get_input_info, train_test_split, tilize_inputs]')
    parser.add_argument('-t', '--tile_size', default=TILE_SIZE, help='tile size INT')
    args = parser.parse_args()
    print (f'{args.stage, args.dataset, args.operation, args.tile_size}')
    process_args(args)
    
