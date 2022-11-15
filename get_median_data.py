import json
import os
import numpy as np
import rasterio
import math
from pathlib import Path
import argparse
from config import *

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def preprocess_points(points):
    '''Ensure that we have top-left and bottom-right coordinates of the legend'''
    # Get the outermost points if given coordinates form a polygon
    if len(points) > 2:
        points = bounding_box(points)

    # Sort points by x-axis
    points = sorted(points, key=lambda x : x[0])

    # Swap y point axis if needed
    # [0, 1        [1, 0]
    #  1, 0]   --> [0, 1]
    if points[0][1] > points[1][1]:
        points[0][1], points[1][1] = points[1][1], points[0][1]

    points = [(int(point[0]), int(point[1]))for point in points]
    return points

def main(args):

    data_path = os.path.join(CHALLENGE_INP_DIR, args.stage)
    legend_median_json_path = 'legends_median_data.json'

    if os.path.exists(legend_median_json_path):
        with open(legend_median_json_path) as fp:
            median_data = json.load(fp)
    else:
        median_data = {}

    json_paths = []

    img_paths = []
    for root, _, files in os.walk(data_path, topdown=False):
        for file in files:
            if '.json' in file:
                json_path = os.path.join(root, file)
                json_paths.append(json_path)
                img_paths.append(json_path.replace('.json', '.tif'))

    for ix, img_path in enumerate(img_paths):

        img_name = Path(img_path).stem
        print(f"{ix}/{len(img_paths)} : {img_name}")

        # AZ_PrescottNF_basemap.json doesn't have the corresponding input image
        # skip such cases
        if not os.path.exists(img_path):
            continue

        #Read the raw image
        with rasterio.open(img_path, 'r') as ds:
            data = ds.read()
        #transform from channel first to channel last
        data = np.moveaxis(data, 0, -1)

        #Get the legend position
        json_path = img_path.replace('.tif', '.json')
        with open(json_path) as f:
            legend_data = json.load(f)

        for shape in legend_data['shapes']:

            suffix = shape['label']        
            points = shape['points']

            if 'poly' not in suffix:
                continue

            points = preprocess_points(points.copy())

            legend = data[points[0][1]:points[1][1], points[0][0]:points[1][0], :]

            # In some cases, the legend shape doesn't not form a rectangle. Use white as default.
            if math.prod(legend.shape):
                rgb_median = [int(x) for x in np.median(legend, axis=(0,1))]
            else:
                print("Bad legend. Using (255, 255, 255) as median value")
                print(img_name + '_' + suffix)
                rgb_median = [255, 255, 255]
            
            median_data.update({img_name + '_' + suffix : rgb_median})
    
    legends_median_data_path = os.path.join(ROOT_PATH, 'eda/legends_median_data_validation.json')
    with open(legends_median_data_path, 'w') as fp:
        json.dump(median_data, fp, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare inputs parser')
    parser.add_argument('-s', '--stage', default='validation', help='which stage? [training, testing, validation]')
    args = parser.parse_args()
    main(args)