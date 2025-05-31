#!/bin/bash

# NetFound Bruteforce Detection Finetuning Script
# This script finetunes the snlucsb/netFound-640M-base model on your bruteforce detection data
# Supports multi-GPU training

set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

export HF_HOME="$HOME/.cache/huggingface"
export TMPDIR="$HOME/tmp"
export TEMP="$HOME/tmp"
export TMP="$HOME/tmp"

# Create temp directory if it doesn't exist
mkdir -p "$HOME/tmp"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate netfound

# Configuration
MODEL_NAME="snlucsb/netFound-640M-base"
DATA_DIR="data/bruteforce/finetuning/final/combined"
OUTPUT_DIR="models/bruteforce_finetuned"
TENSORBOARD_DIR="runs/bruteforce_finetuning"

cat > /tmp/train_with_fixed_loader.py << 'PY'
import sys, os

# 确保当前工作目录在 Python 路径中
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 保证可以 import 到原训练代码
sys.path.append(os.path.join(current_dir, 'src/train'))

# 1) 导入你刚才创建的修复版数据加载函数
from fix_existing_data import load_bruteforce_datasets_from_combined_fixed as fixed_loader
import utils     # 原 utils.py

# 2) Monkey-patch：替换 utils.load_train_test_datasets
utils.load_train_test_datasets = fixed_loader

# 3) 调用原始 NetfoundFinetuning 的 main
from NetfoundFinetuning import main
if __name__ == "__main__":
    main()
PY

# Multi-GPU configuration
export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=1

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

echo "Starting NetFound Bruteforce Detection Finetuning..."
echo "Model: $MODEL_NAME"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Multi-GPU training with torchrun
torchrun --nproc_per_node=$NUM_GPUS \
    /tmp/train_with_fixed_loader.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_train \
    --do_eval \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --logging_dir "$TENSORBOARD_DIR" \
    --report_to tensorboard \
    --problem_type single_label_classification \
    --num_labels 2 \
    --validation_split_percentage 20 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --fp16

echo "Finetuning completed! Model saved to: $OUTPUT_DIR"
echo "View training progress with: tensorboard --logdir $TENSORBOARD_DIR"
