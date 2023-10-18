# mlops_2
Repository for diamonds' price prediction using `KNeighborsRegressor` from `sklearn.neighbors`.

Raw data was taken from [Kaggle competition](https://www.kaggle.com/datasets/shivam2503/diamonds).

Price prediction is performed as `dvc pipeline` with 7 stages:

```
+---------------------------+  
| data/raw/diamonds.csv.dvc |  
+---------------------------+  
              *                       
        +----------+           
        | drop-dup |           # Stage 1
        +----------+           
              *                
         +---------+           
         | drop-na |           # Stage 2
         +---------+           
              *                 
      +---------------+        
      | drop-outliers |        # Stage 3
      +---------------+        
              *                
      +---------------+        
      | preprocessors |        # Stage 4
      +---------------+        
              *                     
          +-------+            
          | split |            # Stage 5
          +-------+            
         **        **          
       **            *         
      *               **       
+-------+               *      
| train |             **       # Stage 6
+-------+            *         
         **        **          
           **    **            
             *  *              
        +----------+           
        | evaluate |           # Stage 7
        +----------+           
```
## Instalation
```
conda create "python>=3.8,<3.11" -n diamonds-dvc
```
```
conda activate diamonds-dvc
```
```
python -m pip install -r requirements.txt (or requirements-dev.txt)
```
## Getting raw data
To get raw diamonds.csv you need to provide your Kaggle-token (follow [Kaggle Public API instructions](https://www.kaggle.com/docs/api). 

Command for raw data download:
```
python scripts/data_scripts/get_data.py
```
## Dvc pipeline
```
dvc repro
```
## !/bin/bash pipeline
```
source utils/pipeline.sh
```
## Project structure
```console
.
├── data
│   ├── raw
│   │   ├── diamonds.csv
│   │   └── diamonds.csv.dvc
│   ├── stage1
│   │   └── diamonds.csv
│   ├── stage2
│   │   └── diamonds.csv
│   ├── stage3
│   │   └── diamonds.csv
│   ├── stage4
│   │   └── diamonds.csv
│   └── stage5
│       ├── test.csv
│       └── train.csv
├── dvc.lock
├── dvc.yaml
├── evaluate
│   └── scores_extended.json
├── models
│   └── model_knr.pkl
├── params.yaml
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── scripts
│   ├── data_scripts
│   │   ├── drop_dup.py
│   │   ├── drop_na.py
│   │   ├── drop_outliers.py
│   │   ├── get_data.py
│   │   ├── preprocessors.py
│   │   ├── train_test_split.py
│   │   └── utils.py
│   └── model_scripts
│       ├── evaluate.py
│       ├── model_knr.py
│       └── utils.py
├── setup.cfg
└── utils
    ├── commands.txt
    └── pipeline.sh
```
