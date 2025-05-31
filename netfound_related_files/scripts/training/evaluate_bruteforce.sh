#!/bin/bash

# NetFound Bruteforce Detection Evaluation Script
# This script evaluates a finetuned model on your bruteforce detection data

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate netfound

# Configuration - modify these paths as needed
MODEL_PATH="models/bruteforce_finetuned"  # Path to your finetuned model
DATA_DIR="data/bruteforce/finetuning/final/combined"
OUTPUT_DIR="evaluation_results/bruteforce"
TENSORBOARD_DIR="runs/bruteforce_evaluation"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

echo "Starting NetFound Bruteforce Detection Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python src/train/NetfoundFinetuning.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_eval \
    --per_device_eval_batch_size 32 \
    --logging_dir "$TENSORBOARD_DIR" \
    --report_to tensorboard \
    --problem_type single_label_classification \
    --num_labels 2 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 32 \
    --streaming \
    --fp16 \
    --dataloader_pin_memory

echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
echo "View evaluation metrics with: tensorboard --logdir $TENSORBOARD_DIR" 