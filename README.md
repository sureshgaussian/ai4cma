# Introduction
This repo contains the code for the solution proposed by Gaussian Solutions LLC for the DARPA's challenge 'AI for Critical Mineral Assessment Competition'. More about the competition can be read [here] https://criticalminerals.darpa.mil/Leaderboard

# Approach

Please refer the [Technical brief](https://docs.google.com/document/d/14Oicvt6i5xTHAOS9gkVIVMF5okK_hokVzQxuZGgg5xc/edit?usp=sharing) for an overview of the approach. 

## Setting up the environment
```
conda create -n env_cma python=3.9
conda activate env_cma
git clone https://github.com/sureshgaussian/ai4cma.git
cd ai4cma
pip install -r requirements.txt
```
## Preparing inputs
All the parameters required to process the inputs, run the experiments, generate predictions are defined in `config.py` file.
<br >
Specify the `ROOT_PATH` in `config.py`. This directory is used to hold the raw and preprocessed data, model checkpoints, and results.<br >
Extract the training and validation data to `$ROOT_PATH/data/`
```
DATA_ROOT 
├── data
│   ├── training
│   ├── validation
```
Run the below script to generate the input descriptors and tiled inputs. 
```
python prepare.py -d challenge -o prepare_inputs
```
This creates a folder structure as shown below
```
DATA_ROOT    
├── tiled_inputs
│   ├── info
│   │   ├── challenge_training_files.csv (holds meta data of each binary raster file)
│   │   ├── challenge_testing_files.csv
│   ├── challenge_training
│   │   ├── info
│   │   │   ├── all_tiles.csv (holds the tile information)
│   │   │   ├── balanced_tiles.csv
│   │   ├── inputs
│   │   ├── legends
│   │   ├── masks
│   ├── challenge_testing
│   │   ├── ...
```
## Training
Parameters related to training are specified in `$PROJECT_DIR/config.py`.
```
python train.py -d challenge
```
Saves the model checkpoints under `$PROJECT_DIR/tmp`.

## Inference
Download the trained models from [here](https://drive.google.com/drive/folders/1LycmdhAzBmzk6C3I_6GbvfoXyIyK_U-1?usp=share_link) and place them under `$PROJECT_DIR/temp`.
Run the script to generate tiles for each validation file, run the inference and save the stitched predictions under `$ROOT_PATH/results/`
Note: For any inference task, please place the files under 'validation' folder for the script to pick up the files. This is a limitation of how we coded the path set up, relative paths etc, and not the algorithm limitation.
```
python inference.py -d challenge -s validation 
```
```
DATA_ROOT  
├── results
│   ├── testing
│   ├── validation
│   │   ├── ArthurTaylor_1990_Fig28_contour_line.tif
│   │   ├── pp1410b_Lower_Mesozoic_diaba_pt.tif
│   │   .
│   │   .
```

## Postprocessing
`Step-1` of post processing to remove false positives within the map region. Erodes the line predictions
```
python postprocessing.py
```
```
DATA_ROOT  
├── results_pp_within_map
│   ├── testing
│   ├── validation
│   │   ├── ArthurTaylor_1990_Fig28_contour_line.tif
│   │   ├── pp1410b_Lower_Mesozoic_diaba_pt.tif
│   │   .
│   │   .
```
`Step-2` of post-processing to remove false positives outside the map region.
```
python filter_non_map_region/inference_map.py
```
```
python postprocessing_submission.py
```
```
DATA_ROOT  
├── results_pp_outside_map
│   ├── testing
│   ├── validation
│   │   ├── ArthurTaylor_1990_Fig28_contour_line.tif
│   │   ├── pp1410b_Lower_Mesozoic_diaba_pt.tif
│   │   .
│   │   .
```
## Visualize results
Run the below command for qualitative analysis of predictions. Sample visualizations are available in `$PROJECT_DIR/sample_viz`.
```
python generate_visualizations.py 
```
```
DATA_ROOT  
├── results_contours
```
