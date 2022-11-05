import json
import os
import numpy as np
import rasterio
import cv2
import math
from sklearn.metrics import f1_score
from scipy import ndimage
from pathlib import Path

data_path = '/home/suresh/challenges/ai4cma/data'
legend_median_json_path = 'legends_median_data.json'

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def preprocess_points(points):
    '''Ensure that we have top-left and bottom-right coordinates of the legend'''
    # Get the outrmost points if coordinates of a polygon are given
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

def main():


    if os.path.exists(legend_median_json_path):
        with open(legend_median_json_path) as fp:
            median_data = json.load(fp)
    else:
        median_data = {}
    bad_legends = {}
    #Gather all the image names
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
            
            # print(points)
            # # [[0, 8124], [43, 8164]]

            legend = data[points[0][1]:points[1][1], points[0][0]:points[1][0], :]

            # In some cases, the legend shapes do not form a rectangle. Ignore them. 
            if math.prod(legend.shape) == 0:
                print("Bad legend")
                print(img_name + '_' + suffix)
                bad_legends.update({img_name + '_' + suffix : legend.shape})
                continue
            # print(legend.shape)
            # cv2.imshow('legend', legend)
            # cv2.waitKey(0)

            # Read seg to validate the rgb values
            # seg_path = img_path.split('.')[0] + '_' + suffix + '.tif'
            # print(seg_path)
            # with rasterio.open(seg_path, 'r') as ds:
            #     seg = ds.read()
            # #transform from channel first to channel last
            # seg = np.moveaxis(seg, 0, -1)
            # seg = seg[:,:,0]

            if 'poly' in suffix:
                
                #Try masking the image with median pixel of the legend. Polygon matching.
                # cv2.imshow('legend', legend)
                # cv2.waitKey(0)
                rgb_median = [int(x) for x in np.median(legend, axis=(0,1))]
                # print(rgb_median)
                median_data.update({img_name + '_' + suffix : rgb_median})

        # if ix > 5:
        #     break
    
    legends_median_data_path = f'legends_median_data.json'
    with open(legends_median_data_path, 'w') as fp:
        json.dump(median_data, fp, indent=4)

    bad_legends_path = 'bad_legends_training.json'
    with open(bad_legends_path, 'w') as fp:
        json.dump(bad_legends, fp, indent=4)
    print(bad_legends)

def get_legend_median(img_path, label):
    '''
    img_path : abs path to image
    '''
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
        if shape['label'] == label:
            points = shape['points']

    shape = [shape for shape in legend_data['shapes'] if shape['label'] == label]

if __name__ == '__main__':
    main()