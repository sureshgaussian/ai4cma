# ai4cma
Preparing inputs
```
python prepare.py -d challenge -o prepare_inputs
```
Creates a folder structure 
```
DATA_ROOT    
├── tiled_inputs
│   ├── info
│   │   ├── challenge_training_files.csv
│   │   ├── challenge_testing_files.csv
│   ├── challenge_training
│   │   ├── info
│   │   │   ├── all_tiles.csv
│   │   │   ├── balanced_tiles.csv
│   │   ├── inputs
│   │   ├── legends
│   │   ├── masks
│   ├── challenge_testing
│   │   ├── ...
├── data
│   ├── training
│   ├── validation
├── results
│   ├── testing
│   ├── validation
│   │   ├── ArthurTaylor_1990_Fig28_contour_line.tif
│   │   ├── pp1410b_Lower_Mesozoic_diaba_pt.tif
│   │   .
│   │   .
├── results_contours

```

Training
```
cd 
python train.py -d challenge
```

Inference
```
python inference.py -d challenge -s validation 
```
