#!/bin/bash
# Generate expert trajectories (buffers) for TM training
# Must be run before stage 1 (pool_tm.py)

cd ..

# ============ Configuration ============
CUDA_ID=0
DATASET="ImageNet"
SUBSET="imagenette"
MODEL="ConvNetD5"
DATA_PATH="/path/to/imagenet"
BUFFER_PATH="./buffers"

TRAIN_EPOCHS=50
NUM_EXPERTS=100
# =======================================

mkdir -p ${BUFFER_PATH}

export CUDA_VISIBLE_DEVICES=${CUDA_ID}
nohup python -u buffer.py \
    --dataset=${DATASET} \
    --subset=${SUBSET} \
    --model=${MODEL} \
    --data_path=${DATA_PATH} \
    --buffer_path=${BUFFER_PATH} \
    --train_epochs=${TRAIN_EPOCHS} \
    --num_experts=${NUM_EXPERTS} \
    > "${BUFFER_PATH}/${SUBSET}_buffer.log" 2>&1 &

echo "Log file: ${BUFFER_PATH}/${SUBSET}_buffer.log"
echo "Process ID: $!"
