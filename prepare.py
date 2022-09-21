import csv 
import os
import glob
from posixpath import splitext
import img2tiles
import json

def prepare_inputs(input_dir, output_dir, tile_size):
    jfiles_path = os.path.join(input_dir, "*.json")
    print(jfiles_path)
    json_files = glob.glob(jfiles_path)
    inputs_descriptor = []
    for json_file in json_files:

        # find corresponding input image
        print(json_file)
        input_file = json_file.replace(".json",".tif")
        in_fname = os.path.split(input_file)[-1]
        print(in_fname)
        in_tiles = img2tiles.split_image_into_tiles(input_dir, in_fname, output_dir, tile_size)

        # prepare the label_pattern_tile for all labels
        j_fname = os.path.split(json_file)[-1]
        #label_patterns = img2tiles.make_label_images(input_dir, in_fname, j_fname, output_dir, tile_size)


        with open(json_file) as jfile:
             label_info = json.load(jfile)

        # split the label_masks into tiles
        for shape in label_info["shapes"]:
            label = shape["label"]
            points = shape["points"]
            label_fname = os.path.splitext(j_fname)[0]+"_"+label+".tif"
            label_mask_tiles = img2tiles.split_image_into_tiles(input_dir, label_fname, output_dir, tile_size)

            # for each label, get the label_pattern file
            label_pattern_fname = img2tiles.make_label_pattern(input_dir, in_fname, label, points, output_dir, tile_size)

            assert(len(label_mask_tiles) == len(in_tiles))
            for idx, tile in enumerate(label_mask_tiles):
                inputs_descriptor.append([in_fname, j_fname, in_tiles[idx], label_pattern_fname, label_mask_tiles[idx]])

            print("Current length of inputs: ", len(inputs_descriptor))
            break
    
    return inputs_descriptor




if __name__ == '__main__':
    tile_size = 256
    input_descriptors = prepare_inputs("./inp", "/home/suresh/challenges/ai4cma/utils/temp", tile_size)
    print(type(input_descriptors))
    with open("inputs.csv", "w") as csv_file:
        write = csv.writer(csv_file)
        write.writerow(["input_file", "json_file", "input_tile", "label_pattern", "label_tile"])
        write.writerows(input_descriptors)

