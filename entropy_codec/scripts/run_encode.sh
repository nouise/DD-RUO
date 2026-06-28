#!/bin/bash
# entropy_codec — 编码：pool.pt → bitstream（ARM 因果 + constriction 流式 CABAC）
# 用法：修改下方配置后 bash scripts/run_encode.sh
# 依赖：torchac + constriction。见 README.md。

cd "$(dirname "$0")/.."

# ============ 配置（按需修改） ============
POOL_PATH="/path/to/pool.pt"
SLICE_INDEX=0            # 取第几个 slice
LAYERS_V="v5"            # 必须与 pool 训练参数一致
ARM_DIM=32
N_HIDDEN_ARM=4
MSE_ERR=0.0000005        # quantize_net 的 MSE 阈值
MASK_SIZE=9              # 必须与 pool 训练时的 mask_size 一致
LAPLACE_RANGE=60         # constriction QuantizedLaplace 支撑区间 [-60,60]，需 > |sent|.max()
CUDA_ID=6
# =========================================

export CUDA_VISIBLE_DEVICES=${CUDA_ID}
LOG_FILE="bitstream_out/encode.log"
mkdir -p bitstream_out

# 配置通过环境变量传给 encode_v2.py（encode_v2.py 读取这些环境变量覆盖默认常量）
python -u encode_v2.py \
    --pool_path "${POOL_PATH}" \
    --slice_index ${SLICE_INDEX} \
    --layers_v ${LAYERS_V} \
    --arm_dim ${ARM_DIM} \
    --n_hidden_arm ${N_HIDDEN_ARM} \
    --mse_err ${MSE_ERR} \
    --mask_size ${MASK_SIZE} \
    --laplace_range ${LAPLACE_RANGE} \
    2>&1 | tee ${LOG_FILE}
