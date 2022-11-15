'''
Scripts to do some post-processing
'''
import cv2
import numpy as np
from utils import preprocess_points, load_legend_data, draw_contours_big
from utils_show import imshow_r, to_grayscale, to_rgb
from glob import glob
import os
from config import *
import pandas as pd

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

        # print(f"num remaining points : {len(remaining_points)}")
        
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
    # print(f"sem_without_legend : {sem_without_legend.shape}")
    # print(f"sem_without_legend : {np.unique(sem_without_legend, return_counts=True)}")
    try:
        sem_without_legend = to_grayscale(sem_without_legend)/255.0
    except:
        sem_without_legend = sem_without_legend/255.0

    return aux_white

def generate_legend_bboxes_masks(step, debug = False):
    '''
    For each poly raster file, generate a mask with 1 in the area of legend bbox and 0 as background
    These shall be used to remove false positives that are seen WITHIN the map. Usually due to close colors
    '''
    csv_path = os.path.join(TILED_INP_DIR, INFO_DIR, f"challenge_{step}_files.csv")
    df = pd.read_csv(csv_path)
    df_poly = df[df['legend_type'] == 'poly']
    df_poly['points_npy'] = df_poly['points'].apply(eval).apply(np.array)

    for name, group in df_poly.groupby('inp_fname'):
        ww, hh = group.iloc[0]['width'], group.iloc[0]['height']
        aux_all = np.zeros((hh, ww), dtype='uint8')

        for ind, row in group.iterrows():
            print(name, ind, row['mask_fname'])
            pts = row['points_npy']
            pts = preprocess_points(pts)
            aux_mask = np.zeros((hh, ww), dtype='uint8')
            cv2.rectangle(aux_mask, pts[0], pts[1], 1, -1)
            if debug:
                imshow_r('single', aux_mask, True)
            aux_all = cv2.bitwise_or(aux_all.copy(), aux_mask)

            mask_name = row['mask_fname'].split('.')[0]
            single_legend_save_name = os.path.join(LEGEND_BBOXES_MASKS_DIR, step, mask_name + '.png')
            cv2.imwrite(single_legend_save_name, aux_mask)

        img_name = name.replace('.tif', '.png')
        all_legends_save_name = os.path.join(LEGEND_BBOXES_MASKS_DIR, step, img_name)
        cv2.imwrite(all_legends_save_name, aux_all)


def remove_false_positives_within_map(step, debug = False):
    '''
    Step 1 of post processing. Try to get rid of false positives WITHIN the area of map
    '''

    legend_bboxes_masks_dir = os.path.join(LEGEND_BBOXES_MASKS_DIR, step)
    preds_dir = os.path.join(RESULTS_DIR, step)

    csv_path = os.path.join(TILED_INP_DIR, INFO_DIR, f"challenge_{step}_files.csv")
    df = pd.read_csv(csv_path)
    df_poly = df[df['legend_type'] == 'poly']
    df_poly['points_npy'] = df_poly['points'].apply(eval).apply(np.array)

    for name, group in df_poly.groupby('inp_fname'):
        print(name)
        # if 'AR_Buffalo' in name:
        #     continue 
        raw_img_path = os.path.join(CHALLENGE_INP_DIR, 'training' if step == 'testing' else step, name)
        raw_img = cv2.imread(raw_img_path)

        all_legends_path = os.path.join(legend_bboxes_masks_dir, name.replace('.tif', '.png'))
        # all_legends = draw_contours_big(raw_img_path, all_legends_path)
        # all_legends = cv2.imread(all_legends_path)

        for ind, row in group.iterrows():
            print(ind, row['mask_fname'])
            # if 'AR_Buffalo_west_Mbs_poly.tif' not in row['mask_fname']:
            #     continue
            raw_pred_path = os.path.join(preds_dir, row['mask_fname'])
            target_path = os.path.join(CHALLENGE_INP_DIR, 'training', row['mask_fname'])
            # target = cv2.imread(target_path)
            raw_pred = draw_contours_big(raw_img_path, raw_pred_path, target_path)

            pos_legend_path = os.path.join(legend_bboxes_masks_dir, row['mask_fname'].replace('.tif','.png'))
            # pos_legend_gray = cv2.imread(pos_legend_path, 0)
            pos_legend = draw_contours_big(raw_img_path, pos_legend_path)

            raw_pred_gray = cv2.imread(raw_pred_path, 0)
            all_legends_gray = cv2.imread(all_legends_path, 0)

            predicted_legends = cv2.bitwise_and(all_legends_gray, raw_pred_gray)
            # predicted_legends = cv2.morphologyEx(predicted_legends, cv2.MORPH_OPEN, (15,15))
            # predicted_legends = draw_contours_big(raw_img_path, predicted_legends)

            pts = row['points_npy']
            pts = preprocess_points(pts)
            patch_rgb = raw_img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0], :]
            rgb_median = tuple([int(x) for x in np.median(patch_rgb, axis=(0,1))])

            lower = upper = rgb_median

            print(f"positive rgb : {rgb_median}")
            img_pos = cv2.inRange(raw_img, lower, upper)           
            print(img_pos.shape)
            
            # img_pos = draw_contours_big(raw_img, raw_pred, target_path)
            if debug:
                imshow_r(f"raw_pred_{row['mask_fname']}", raw_pred, True)

            neg_boxes = group[group['mask_fname'] != row['mask_fname']]
            pred_final = raw_pred_gray.copy()
            for indd, other_row in neg_boxes.iterrows():
                pts = other_row['points_npy']
                pts = preprocess_points(pts)

                patch = predicted_legends[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
                onez = np.argwhere(patch == 1)
                if len(onez):
                    print('fp', other_row['mask_fname'])

                    # Get rgb value of the patch
                    patch_rgb = raw_img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0], :]
                    rgb_median = tuple([int(x) for x in np.median(patch_rgb, axis=(0,1))])
                    lower = upper = rgb_median
                    fp_pos = cv2.inRange(raw_img, lower, upper)

                    fp_pos_aux = np.zeros(fp_pos.shape, dtype='uint8')

                    # Dilate to close the gaps
                    fp_pos = cv2.dilate(fp_pos, np.ones((3, 3), dtype='uint8'),iterations = 10)

                    # Use convex hull may be? But Beware! that might also remove true-positives.
                    # contours, hierarchy = cv2.findContours(fp_pos, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # for i in range(len(contours)):
                    #     hull = cv2.convexHull(contours[i])
                    #     cv2.drawContours(fp_pos_aux, [hull], -1, 255, -1)
                    # fp_pos = fp_pos_aux

                    # imshow_r('fp_pos', fp_pos, True)
                    
                    fp_pos = fp_pos/255
                    fp_pos = fp_pos.astype('uint8')
                    
                    print(fp_pos.shape, pred_final.shape)
                    print("min-max", np.min(pred_final), np.max(pred_final))
                    print("min-max", np.min(fp_pos), np.max(fp_pos))
                    pred_final[fp_pos == 1] = 0
                    fp_pos = draw_contours_big(raw_img, fp_pos, target_path)
                    print(f"fp median {rgb_median}")
                    if debug:
                        imshow_r(f"for {row['mask_fname']}_fp_from_{other_row['mask_fname']}", fp_pos, True)
                    
                    # AR_Buffalo_west_Mf_poly.tif (there are still lots of fps left. Need to check what's happening)
                    # Same with AR_Buffalo_west_Mfw_poly.tif, AR_Jasper_Mfw_poly.tif
            
            save_path = os.path.join(POSTP_INMAP_DIR, step, row['mask_fname'].split('.')[0] + '.png')
            cv2.imwrite(save_path, pred_final)
            pred_final = draw_contours_big(raw_img_path, pred_final, target_path)

            if debug:
                imshow_r('pred_final', pred_final, True)
                cv2.destroyAllWindows()

            # break
        
        # break

def refine_poly_predictions():
    # Get the polygon mask and refine the boudaries based on pixel color.
    pass

def refine_line_predictions():
    # Instead of just erosion, find the line in the original image and see if you can find a line over there. 
    pass

def post_process_lines(step):

    pred_paths_all = glob(os.path.join(RESULTS_DIR, step, '*.tif'))
    pred_paths_lines = [pred_path for pred_path in pred_paths_all if '_line.tif' in pred_path]
    print(pred_paths_lines)
    for pred_path_line in pred_paths_lines:
        pred_name = os.path.basename(pred_path_line)
        pred = cv2.imread(pred_path_line)
        imshow_r(f'raw_{pred_name}', pred, True)
        pred_eroded = cv2.erode(pred, np.ones((5, 5), np.uint8))
        imshow_r(f'eroded_{pred_name}', pred_eroded, True)


if __name__ == '__main__':

    # input_dir = '/home/suresh/challenges/ai4cma/data/training'
    # for raster_file_path in glob.glob(os.path.join(input_dir, '*.tif')):
    #     print(raster_file_path)
    #     if 'pt' not in raster_file_path:
    #         continue
        
    #     json_path = [x for x in glob.glob(os.path.join(input_dir, '*.json')) if os.path.basename(x).split('.')[0] in raster_file_path][0]

    #     discard_preds_outside_map(json_path, debug=True)
    step = 'testing'
    post_process_lines(step)
    # generate_legend_bboxes_masks(step)
    # remove_false_positives_within_map(step)