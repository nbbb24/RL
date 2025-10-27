#!/bin/bash

data_dir="data/raw/ECG-Expert-QA/ECG_Knowledge_(Basic_Q&A).json"

python prepare_data.py \
    --input "$data_dir" \
    --output_dir "data/processed"

echo "Data preparation complete!"