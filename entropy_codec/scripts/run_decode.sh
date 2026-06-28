#!/bin/bash
# entropy_codec — 解码：bitstream → decoded.pt + 合成图像 PNG
# 用法：修改下方配置后 bash scripts/run_decode.sh
# 注意：逐像素流式解码较慢，调试可设 N_IMAGES=2 只解前几张。
# 依赖：torchac + constriction。见 README.md。

cd "$(dirname "$0")/.."

# ============ 配置（按需修改） ============
BITSTREAM_DIR="./bitstream_out"   # encode 产出的码流目录
OUT_DIR="./decoded_out"
LAPLACE_RANGE=60                  # 必须与 encode 端一致
N_IMAGES=""                       # 空=全量；调试填数字（如 2）只解前 N 张
CUDA_ID=6
# =========================================

export CUDA_VISIBLE_DEVICES=${CUDA_ID}
LOG_FILE="${OUT_DIR}/decode.log"
mkdir -p "${OUT_DIR}"

N_IMAGES_ARG=""
[ -n "${N_IMAGES}" ] && N_IMAGES_ARG="--n_images ${N_IMAGES}"

python -u decode_v2.py \
    --bitstream_dir "${BITSTREAM_DIR}" \
    --out_dir "${OUT_DIR}" \
    --laplace_range ${LAPLACE_RANGE} \
    ${N_IMAGES_ARG} \
    2>&1 | tee ${LOG_FILE}
