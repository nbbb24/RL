#!/bin/bash

data_dir="data/raw/ECG-Expert-QA/ECG_Knowledge_Basic_Q_A.json"

python prepare_data.py \
    --input "$data_dir" \
    --output_dir "data/processed"

echo "Data preparation complete!"