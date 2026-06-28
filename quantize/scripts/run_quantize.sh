#!/bin/bash
# Post-quantization evaluation for distilled datasets (shared across TM/DM/DC).
#
# This script applies post-training quantization to a pre-trained synthesis
# network and evaluates the distilled dataset quality after quantization.
#
# Usage: bash scripts/run_quantize.sh

cd "$(dirname "$0")/.."

# ============ Configuration ============

CUDA_ID=0,1
DATASET="ImageNet"
SUBSET="imagenette"
IPC=102

# TensorPool architecture (must match the pre-trained checkpoint)
LAYERS_V="v5"
ARM=32
DIM=4

# Evaluation
NUM_EVAL=5
SYN_LR=0.00996475014835596
MSE_ERR=0.0000005

# Pre-trained pool checkpoint (modify this to your checkpoint path)
POOL_PATH="path/to/your/pool_checkpoint.pt"

# Output directory
SAVE_DIR="./results/quantize/${DATASET}/${SUBSET}/${IPC}"

# ============ Run ============

export CUDA_VISIBLE_DEVICES=${CUDA_ID}

FLAG="${DATASET}_${SUBSET}_ipc${IPC}_${ARM}_${DIM}"
TIMESTAMP=$(date +"%m%d_%H%M")
LOG_DIR="${SAVE_DIR}/${FLAG}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

nohup python -u quantize_pool.py \
    --dataset ${DATASET} \
    --subset ${SUBSET} \
    --ipc ${IPC} \
    --layers_v ${LAYERS_V} \
    --arm ${ARM} \
    --dim ${DIM} \
    --syn_lr_set ${SYN_LR} \
    --mse_err ${MSE_ERR} \
    --pool_path ${POOL_PATH} \
    --save_path ${SAVE_DIR} \
    --FLAG ${FLAG} \
    --num_eval ${NUM_EVAL} \
    > "${LOG_FILE}" 2>&1 &

echo "Script started. Logging to ${LOG_FILE}"
echo "Process ID: $!"
