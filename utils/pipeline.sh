#!/bin/bash
python scripts/data_scripts/get_data.py
python scripts/data_scripts/drop_dup.py data/raw/diamonds.csv
python scripts/data_scripts/drop_na.py data/stage1/diamonds.csv
python scripts/data_scripts/drop_outliers.py data/stage2/diamonds.csv
python scripts/data_scripts/preprocessors.py data/stage3/diamonds.csv
python scripts/data_scripts/train_test_split.py data/stage4/diamonds.csv
python scripts/model_scripts/model_knr.py data/stage5/train.csv model_knr_1
