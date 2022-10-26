import sys 
from unittest import result
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import random
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import math
import json
from datetime import datetime
import glob
import argparse
def overlap_distance_calculate(mat_true, mat_pred, min_valid_range=.1, parallel_workers=1):
    """
    mat_true, mat_pred: 2d matrices, with 0s and 1s only
    min_valid_range: the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    calculate_distance: when True this will not only calculate overlapping pixels
        but also the distances between nearesttrue and predicted pixels
    """
    
    lowest_dist_pairs=[]
    points_done_pred=set()
    points_done_true=set()

    # first calculate the overlapping pixels
    mat_overlap=mat_pred*mat_true
    for x_true, y_true in tqdm(np.argwhere(mat_overlap==1)):
        lowest_dist_pairs.append((((x_true, y_true), (x_true, y_true)), 0.0)) 
        points_done_true.add((x_true, y_true))
        points_done_pred.add((y_true, x_true))
    print('len(lowest_dist_pairs) by overlapping only:', len(lowest_dist_pairs))
    
    diagonal_length=math.sqrt(math.pow(mat_true.shape[0], 2)+ math.pow(mat_true.shape[1], 2))
    min_valid_range=int((min_valid_range*diagonal_length)/100) # in pixels
    print('calculated pixel min_valid_range:', min_valid_range)

    def nearest_pixels(x_true, y_true):
        result=[]
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange=mat_pred[
         max(x_true-min_valid_range, 0): min(x_true+min_valid_range, mat_true.shape[1]),
            max(y_true-min_valid_range, 0): min(y_true+min_valid_range, mat_true.shape[0])
        ]
        for x_pred_shift, y_pred_shift in np.argwhere(mat_pred_inrange==1):
            y_pred=max(y_true-min_valid_range, 0)+y_pred_shift
            x_pred=max(x_true-min_valid_range, 0)+x_pred_shift
            if (x_pred, y_pred) in points_done_pred:
                continue
            # calculate eucledean distances 
            dist_square=math.pow(x_true-x_pred, 2)+math.pow(y_true-y_pred, 2)
            result.append((((x_true, y_true), (x_pred, y_pred)), dist_square))
        return result

    candidates=[(x_true, y_true) for x_true, y_true in tqdm(np.argwhere(mat_true==1)) if (x_true, y_true) not in points_done_true]
    distances=Parallel(n_jobs=parallel_workers)(delayed(nearest_pixels)(x_true, y_true) for x_true, y_true in tqdm(candidates))
    distances = [item for sublist in distances for item in sublist]

    # sort based on distances
    distances=sorted(distances, key=lambda x: x[1])

    # find the lowest distance pairs
    for ((x_true, y_true), (x_pred, y_pred)), distance in tqdm(distances):
        if ((x_true, y_true) in points_done_true) or ((x_pred, y_pred) in points_done_pred):
            # do not consider a taken point again
            continue
        # normalize all distances by diving by the diagonal length  
        lowest_dist_pairs.append((((x_true, y_true), (x_pred, y_pred)), math.sqrt(float(distance))/diagonal_length)) 
        points_done_true.add((x_true, y_true))
        points_done_pred.add((x_pred, y_pred))
    
    return lowest_dist_pairs

def detect_difficult_pixels(map_image, binary_raster, legend_coor, plot=True, set_false_as='hard', color_range=4, baseline_raster=None):
    """
    map_image: the image array for the map image
    binary_raster: 2D array of any channel (out of 3 present) from the true binary raster image 
    legend_coor: coordinate for the legend feature, from the legend json file
    plot: plots different rasters
    set_false_as: when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """
    
    plt.rcParams["figure.figsize"] = (15,22)
        
    # detect pixels based on color of legend
    if legend_coor is not None:
        print('running baseline...')
        pred_by_color=match_by_color(map_image.copy(), legend_coor, color_range=color_range, plot=plot)
        if plot:
            print('prediction based on color of legend:')
            plt.imshow(pred_by_color)
            plt.show()    
    pred_by_color=(1-pred_by_color).astype(int) # flip, so the unpredicted become hard pixels
    pred_by_color=binary_raster*pred_by_color # keep only the part within the true polygon
    if plot:
        print('flipped color predictions (the non-predicted will be hard pixels), within in the polygon range:')
        plt.imshow(pred_by_color)
        plt.show()
    
    if set_false_as=='hard':
        # the pixels that are not within the true polygon should are deemed hard pixels
        final_hard_pixels=(1-binary_raster)|pred_by_color
    else:
        # the outside pixels will be deemed easy!
        final_hard_pixels=pred_by_color
    
    if plot:
        print('final hard pixels (merged):')
        plt.imshow(final_hard_pixels)
        plt.show()

    return final_hard_pixels


def match_by_color(img, legend_coor, color_range=4, plot=False):
    """
    img: the image array for the map image
    legend_coor: coordinate for the legend feature, from the legend json file
    """
    # get the legend coors and the predominant color
    (x_min, y_min), (x_max, y_max) = legend_coor            
    legend_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    if plot:
        print('legend feature:')
        plt.imshow(legend_img)
        plt.show()
    # take the median of the colors to find the predominant color
    r=int(np.median(legend_img[:,:,0]))
    g=int(np.median(legend_img[:,:,1]))
    b=int(np.median(legend_img[:,:,2]))
    sought_color=[r, g, b]
    # capture the variations of legend color due to scanning errors
    lower = np.array(sought_color)-color_range
    lower[lower<0] = 0
    lower=tuple(lower.tolist())
    upper = np.array(sought_color)+color_range
    upper[upper>255] = 255
    upper=tuple(upper.tolist())
    print('matching the color:', sought_color, 'with color range:', color_range, ', lower:', lower, 'upper:', upper)
    # create a mask to only preserve current legend color in the basemap
    pred_by_color = cv2.inRange(img, lower, upper)/255

    return pred_by_color

def feature_f_score(map_image_path, predicted_raster_path, true_raster_path, legend_json_path=None, min_valid_range=.1,
                      difficult_weight=.7, set_false_as='hard', plot=False, color_range=4, parallel_workers=1):
    
    """
    map_image_path: path to the the actual map image
    predicted_raster_path: path to the the predicted binary raster image 
    true_raster_path: path to the the true binary raster image 
    legend_json_path: (only used for polygons) path to the json containing the coordinates for the corresponding legend (polygon) feature
    min_valid_range: (only used for points and lines) the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    difficult_weight: (only used for polygons) float within [0, 1], weight for the difficlut pixels in the scores (only for polygins)
    set_false_as: (only used for polygons) when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """
    

    true_raster=cv2.imread(true_raster_path)
    true_raster=true_raster[:,:,0]
    
    predicted_raster=cv2.imread(predicted_raster_path)
    if len(predicted_raster.shape)==3:
        predicted_raster=predicted_raster[:,:,0]
    elif len(predicted_raster.shape)==2:
        predicted_raster=predicted_raster
    else:
        print('predicted_raster shape is not 3 or 2!!!')
        raise ValueError
    
    for item in np.unique(predicted_raster):
        if int(item) not in [0, 1, 255]:
            print('value in predicted raster:', int(item), 'not in permissible values:', [0, 1, 255])
            raise ValueError
    
    predicted_raster[predicted_raster==255] = 1
    
    
    extention=os.path.basename(true_raster_path).split('.')[-1]
    
    legend_feature=os.path.basename(true_raster_path).replace(os.path.basename(map_image_path).replace('.'+extention, '')+'_', '').replace('.'+extention, '')
    feature_type=legend_feature.split('_')[-1]
    print('feature type:', feature_type)
    
    start=datetime.now()
    if plot or feature_type =='poly':
        img=cv2.imread(map_image_path)
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('time check 1:', datetime.now()-start)
    
    # plot: overlay the true and predicted values on the map image
    if plot:
        im_copy=img.copy()
        for center in np.argwhere(predicted_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (0,255,0), -1) # green
        print('Predicted raster overlayed on map image:')
        plt.rcParams["figure.figsize"] = (15,22)
        plt.imshow(im_copy)
        plt.show()
        im_copy=img.copy()
        for center in np.argwhere(true_raster==1):
            cv2.circle(im_copy, (center[1], center[0]), 1, (255,0,0), -1) # red
        print('True raster overlayed on map image:')
        plt.imshow(im_copy)
        plt.show()
    
    start=datetime.now()
    legend_coor=None
    print('looking for legend_feature in the json:', legend_feature)
    if legend_json_path is not None:
        legends=json.loads(open(legend_json_path, 'r').read())
        for shape in legends['shapes']:
            if legend_feature ==shape['label']:
                legend_coor=shape['points']
        print('legend_coor:', legend_coor)
    print('time check 2:', datetime.now()-start)
    
    mat_true, mat_pred=true_raster, predicted_raster
                       
    if feature_type in ['line', 'pt']: # for point and lines
        lowest_dist_pairs=overlap_distance_calculate(mat_true, mat_pred,
                                                     min_valid_range=min_valid_range, parallel_workers=parallel_workers)
        print('len(lowest_dist_pairs):', len(lowest_dist_pairs))
        sum_of_similarities=sum([1-item[1] for item in lowest_dist_pairs])
        print('sum_of_similarities:', sum_of_similarities)
        print('num all pixel pred:', len(np.argwhere(mat_pred==1)))
        print('num all pixel true:', len(np.argwhere(mat_true==1)))
        
        num_mat_pred=np.sum(mat_pred) 
        num_mat_true=np.sum(mat_true)
        precision=sum_of_similarities/num_mat_pred if num_mat_pred!=0 else 0.0
        recall=sum_of_similarities/num_mat_true if num_mat_true!=0 else 0.0
        
    else: # for polygon
        
        start=datetime.now()
        overlap=mat_true*mat_pred
        print('time check 3:', datetime.now()-start)

        if difficult_weight is not None:
            
            start=datetime.now()
            difficult_pixels=detect_difficult_pixels(img, true_raster, legend_coor=legend_coor, set_false_as=set_false_as, plot=plot,
                                                    color_range=color_range)
            print('time check 4:', datetime.now()-start)
            
            start=datetime.now()
            difficult_overlap=overlap*difficult_pixels
            num_overlap_difficult=np.sum(difficult_overlap) 
            print('num_overlap_difficult:', num_overlap_difficult)
            num_overlap_easy=np.sum(overlap-difficult_overlap) 
            print('num_overlap_easy:', num_overlap_easy)
            points_from_overlap=(num_overlap_difficult*difficult_weight)+(num_overlap_easy*(1-difficult_weight))
            print('points_from_overlap:', points_from_overlap)
            
            pred_difficult=mat_pred*difficult_pixels
            num_mat_pred_difficult=np.sum(pred_difficult) 
            print('num_mat_pred_difficult:', num_mat_pred_difficult)
            num_mat_pred_easy=np.sum(mat_pred-pred_difficult) 
            print('num_mat_pred_easy:', num_mat_pred_easy)
            total_pred=(num_mat_pred_difficult*difficult_weight)+(num_mat_pred_easy*(1-difficult_weight))
            print('total prediction points contended:', total_pred)
            precision=points_from_overlap/total_pred if total_pred!=0 else 0.0
            
            
            true_difficult=mat_true*difficult_pixels
            num_mat_true_difficult=np.sum(true_difficult) 
            print('num_mat_true_difficult:', num_mat_true_difficult)
            num_mat_true_easy= np.sum(mat_true-true_difficult) 
            print('num_mat_true_easy:', num_mat_true_easy)
            total_true=(num_mat_true_difficult*difficult_weight)+(num_mat_true_easy*(1-difficult_weight))
            print('total true points to be had:', total_true)
            recall=points_from_overlap/total_true if total_true!=0 else 0.0
            print('time check 5:', datetime.now()-start)
        
        else:
            num_overlap=np.sum(overlap) 
            print('num_overlap:', num_overlap)
            num_mat_pred=np.sum(mat_pred)
            print('num_mat_pred:', num_mat_pred)
            num_mat_true=np.sum(mat_true)
            print('num_mat_true:', num_mat_true)

            precision=num_overlap/num_mat_pred if num_mat_pred!=0 else 0.0
            recall=num_overlap/num_mat_true if num_mat_true!=0 else 0.0
        
    
    # calculate f-score
    f_score=(2 * precision * recall) / (precision + recall) if precision+recall!=0 else 0.0

    return precision, recall, f_score


def test_feature_f1():
    map_image_path = "../data/training/AK_Umiat.tif"
    predicted_image = "../results/testing/AK_Umiat_Kn_poly.tif"
    true_raster_image = "../data/training/AK_Umiat_Kn_poly.tif"
    legend_json_path = "../data/training/AK_Umiat.json"
    [precision, recall, f_score] = feature_f_score(map_image_path=map_image_path, predicted_raster_path=predicted_image,
            true_raster_path=true_raster_image, legend_json_path=legend_json_path)
    print(precision, recall, f_score)
    return

def get_info_csv(input_csv, results_dir, inputs_dir, score_csv_file):
    df = pd.read_csv(input_csv)
    results = glob.glob(os.path.join(results_dir, "*.tif"))
    if not results:
        print(f'no files in {results_dir}... Check **** ')
        return
    results_tifs = [os.path.basename(x) for x in results]
    input_tifs = df['mask_fname'].values

    info_list = []
    for results_tif in results_tifs:
        print(results_tif)
        if results_tif in input_tifs:
            inp_fname = df.loc[df['mask_fname'] == results_tif, 'inp_fname'].values[0]
            img_path = os.path.join(inputs_dir, inp_fname)
            predicted_raster_path = os.path.join(results_dir, results_tif)
            original_raster_path = os.path.join(inputs_dir, results_tif)
            legend_json_path = os.path.join(inputs_dir, inp_fname.replace('.tif', '.json'))
            legend_type = results_tif.replace('.tif','').split("_")[-1]
            info_list.append([img_path, predicted_raster_path, original_raster_path, legend_json_path, legend_type])

    info_df = pd.DataFrame(info_list, columns=['input_image_path', 'predicted_image_path', 'original_raster_path', 'legend_json_path', 'legend_type'])
    info_df.to_csv(score_csv_file,index=False)
    return

    

def calculate_score(info_csv, score_out_csv):
    """
    Inputs:
        - csv file with [input_image_path, predicted_image_path, original_raster_path, legend_json_path, legend_type]
    Output:
        - final_score
    """
    df = pd.read_csv(info_csv)
    #print(df.columns)
    #results = df.apply(lambda row: feature_f_score(row["input_image_path"], row["predicted_image_path"], row["original_raster_path"], row["legend_json_path"]), axis=1)
    # TODO: should be a faster way to do the following
    results =  []
    for idx, row in df.iterrows():
        #print(row)
        results.append(feature_f_score(row["input_image_path"], row["predicted_image_path"], row["original_raster_path"], row["legend_json_path"]))
    results_df = pd.DataFrame(results, columns=['precision', 'recall', 'f1_score'])
    #print("results_df columns are: ", results_df.columns) 
    df_results = pd.concat([df, results_df], axis=1)
    df_results.to_csv(score_out_csv, index=False)

    poly_type_group = df_results.groupby('legend_type')
    score_df = poly_type_group.median()
    poly_score = score_df.f1_score.get("poly", 0)
    line_score = score_df.f1_score.get("line", 0)
    pt_score = score_df.f1_score.get("pt", 0)
    
    score = ((2*poly_score) + line_score + pt_score)/4
    print( f'poly = {poly_score} , line = {line_score} , pt = {pt_score}, score = {score}')

    return score

def process_args(args):
    # prepare the required csv for scoring from the args
    input_csv = args.csv_file
    results_dir = args.results_dir
    inputs_dir = args.inputs_dir
    score_csv_file = "temp_score_input.csv"
    get_info_csv(input_csv, results_dir, inputs_dir, score_csv_file)
    # call calculate_score with the csv file
    score_out_csv = args.score_file

    print("Going into scoring calculations.....")
    score = calculate_score(score_csv_file, score_out_csv)
    print(score)
    return 

def test_calculate_score():

    score = calculate_score("test_f1.csv")
    print(score)
    return 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Challenge score parser')
    parser.add_argument('-i', '--inputs_dir', help='input directory where original files are ')
    parser.add_argument('-r', '--results_dir', help='results directory where inference results are written out ')
    parser.add_argument('-c', '--csv_file', help='csv file that describes inference results ')
    parser.add_argument('-s', '--score_file', help='score csv file output with precision, recall, n f1 ')

    args = parser.parse_args()
    print(len(sys.argv))
    if len(sys.argv) != 9:
        print(f'Wrong inputs...')
        exit()
    # prepare_for_submission()
    process_args(args)    
