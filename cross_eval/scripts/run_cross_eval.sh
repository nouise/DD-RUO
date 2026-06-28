#!/bin/bash
# Cross-architecture evaluation for distilled datasets
# Loads synthetic images/labels from stage 2 and evaluates on multiple architectures

cd "$(dirname "$0")/.."

# ============ Configuration ============
CUDA_ID=0
DATASET="ImageNet"
SUBSET="imagenette"
IPC=102
DATA_PATH="/path/to/imagenet"

# Path to synthetic images/labels (.pt files from stage 2 quantization)
# phase 1 = post-quantization, mse threshold used in stage 2
IMAGES_PATH="./results/quantize/${DATASET}/${SUBSET}/${IPC}/${DATASET}_${SUBSET}_ipc${IPC}_32_4/images_1_5e-07.pt"
LABELS_PATH="./results/quantize/${DATASET}/${SUBSET}/${IPC}/${DATASET}_${SUBSET}_ipc${IPC}_32_4/labels_1_5e-07.pt"

# Evaluation settings
MODELS="ResNet18ImageNet,VGG11,AlexNet,ViT"
NUM_EVAL=5
EPOCH_EVAL=1000
BATCH_TRAIN=256
BATCH_REAL=256

SAVE_DIR="./results/cross_eval/${SUBSET}"
# =======================================

if [ -z "$IMAGES_PATH" ] || [ -z "$LABELS_PATH" ]; then
    echo "Error: IMAGES_PATH and LABELS_PATH must be set"
    exit 1
fi

mkdir -p ${SAVE_DIR}
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${SAVE_DIR}/cross_eval_${timestamp}.log"

export CUDA_VISIBLE_DEVICES=${CUDA_ID}
nohup python -u cross_evaluate.py \
    --images_path ${IMAGES_PATH} \
    --labels_path ${LABELS_PATH} \
    --dataset ${DATASET} \
    --subset ${SUBSET} \
    --data_path ${DATA_PATH} \
    --models ${MODELS} \
    --num_eval ${NUM_EVAL} \
    --epoch_eval_train ${EPOCH_EVAL} \
    --batch_train ${BATCH_TRAIN} \
    --batch_real ${BATCH_REAL} \
    --save_dir ${SAVE_DIR} \
    > ${log_file} 2>&1 &

echo "Log file: ${log_file}"
echo "Process ID: $!"
