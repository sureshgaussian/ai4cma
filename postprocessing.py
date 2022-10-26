'''
Scripts to do some post-processing
'''
import cv2
import numpy as np
from utils import preprocess_points, load_legend_data
from utils_show import imshow_r, to_grayscale, to_rgb
import glob
import os

def discard_preds_outside_map(legend_json_path, debug = False):
    '''
    Exclude predictions from outside the map region. This reduces false positives
    '''
    img_path = legend_json_path.replace('.json', '.tif')
    img = cv2.imread(img_path)
    if debug:
        imshow_r(os.path.basename(img_path), img)
    #First check where the legends are located.
    # imshow_r('raw_pred', pred)
    
    # Plot all legends on an empty image
    points_all = []
    legend_data = load_legend_data(legend_json_path)

    img_h = legend_data['imageHeight']
    img_w = legend_data['imageWidth']
    aux = np.zeros((img_h, img_w), dtype='uint8')
    if debug:
        # For sanity check. Load a sample raster file (say a polygon). 
        # The polygon region should remain unchanged in the final output. 
        try:
            sample_suffix = [shape['label'] for shape in legend_data['shapes'] if 'poly' in shape['label']][1]
        except:
            sample_suffix = legend_data['shapes'][0]['label']
        sem_path = f"{legend_json_path.split('.')[0]}_{sample_suffix}.tif"
        aux = cv2.imread(sem_path, 0)*255
        print(aux.shape)
        print(f"aux : {np.unique(aux, return_counts=True)} shape : {aux.shape}")

        # If raster file is in rgb format, check if all the channels have same values
        if aux.shape[-1] == 3:
            assert np.logical_and((aux[:,:,0]==aux[:,:,1]).all(), (aux[:,:,1]==aux[:,:,2]).all())
        imshow_r('aux_raster', aux)

    # Draw bounding boxes of legends
    for shape in legend_data['shapes']:

        suffix = shape['label']     
        points = shape['points']

        points = preprocess_points(points)
        points_all.append(points)

        aux = cv2.rectangle(aux, points[0], points[1], 255, 20)

    if debug:
        imshow_r('legends', aux)

    # Discard the points right of y_min and below x_min.
    # Do this until there are no legends left in the image. 
    sem_without_legend = aux.copy()
    remaining_points = points_all.copy()
    aux_white = np.ones((img_h, img_w), dtype='uint8')
    while len(remaining_points):

        print(f"num remaining points : {len(remaining_points)}")
        
        # After preprocessing, legend is ordered as top-left, bottom-right corners. 
        legend_x_min = min(remaining_points, key=lambda point: point[0][0])
        legend_y_min = min(remaining_points, key=lambda point: point[0][1])

        # We only need top left corner for processing
        point_x_min = legend_x_min[0]
        point_y_min = legend_y_min[0]
        print(f"point_x_min : {point_x_min} point_y_min : {point_y_min}")

        if debug:
            # Check which points are picked
            selected_points_rgb = to_rgb(sem_without_legend)
            for pt in remaining_points:
                cv2.circle(selected_points_rgb, pt[0], 5, (0, 255, 0), 50, -1)
            cv2.circle(selected_points_rgb, point_x_min, 5, (255, 0, 0), 50, -1)
            cv2.circle(selected_points_rgb, point_y_min, 5, (0, 0, 255), 50, -1)
            imshow_r('selected points', selected_points_rgb)

        # Discard the area to the right of y_min and blow x_min
        if debug:
            imshow_r('step before removing legend', sem_without_legend)
        pad = 10
        sem_without_legend[point_x_min[1] - pad:,point_x_min[0] - pad:] = 0
        sem_without_legend[point_y_min[1] - pad:,point_y_min[0] - pad:] = 0
        aux_white[point_x_min[1] - pad:,point_x_min[0] - pad:] = 0
        aux_white[point_y_min[1] - pad:,point_y_min[0] - pad:] = 0
        if debug:
            imshow_r('step after removing legend', sem_without_legend, True)
            imshow_r('aux_white', aux_white*255, True)

        # Remove these points from the list and repeat until no legends are left.
        # This is to handle case when legends fall between x_min and y_min. 
        points_to_remove = []
        points_to_remove.extend([point for point in remaining_points if point[0][0] >= point_x_min[0] and point[0][1] >= point_x_min[1]])
        points_to_remove.extend([point for point in remaining_points if point[0][0] >= point_y_min[0] and point[0][1] >= point_y_min[1]])
        print(f"num points to remove {len(points_to_remove)}")

        remaining_points = [point for point in remaining_points if point not in points_to_remove]
    
    if debug:
        imshow_r(os.path.basename(legend_json_path), sem_without_legend, True)
        # cv2.destroyAllWindows()

    # Rescale the image intensity
    print(f"sem_without_legend : {sem_without_legend.shape}")
    print(f"sem_without_legend : {np.unique(sem_without_legend, return_counts=True)}")
    try:
        sem_without_legend = to_grayscale(sem_without_legend)/255.0
    except:
        sem_without_legend = sem_without_legend/255.0

    return aux_white
    
if __name__ == '__main__':

    input_dir = '/home/suresh/challenges/ai4cma/data/training'
    for raster_file_path in glob.glob(os.path.join(input_dir, '*.tif')):
        print(raster_file_path)
        if 'pt' not in raster_file_path:
            continue
        
        json_path = [x for x in glob.glob(os.path.join(input_dir, '*.json')) if os.path.basename(x).split('.')[0] in raster_file_path][0]

        discard_preds_outside_map(json_path, debug=True)
