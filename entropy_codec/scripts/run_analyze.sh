#!/bin/bash
# entropy_codec — 分析：原 pool vs 码流 vs 解码结果（bpp / PSNR / max_abs_diff / 对比图）
# 用法：修改下方配置后 bash scripts/run_analyze.sh
# 前置：已跑过 run_encode.sh（产出 bitstream_out/）；若 decoded_out/decoded.pt 不存在会自动解码一次。
# 依赖：torchac + constriction。见 README.md。

cd "$(dirname "$0")/.."

# ============ 配置（按需修改） ============
POOL_PATH="/path/to/pool.pt"
BITSTREAM_DIR="./bitstream_out"
ANALYSIS_DIR="./analysis_out"
CUDA_ID=6
# =========================================

export CUDA_VISIBLE_DEVICES=${CUDA_ID}
LOG_FILE="${ANALYSIS_DIR}/analyze.log"
mkdir -p "${ANALYSIS_DIR}"

python -u analyze.py \
    --pool_path "${POOL_PATH}" \
    --bitstream_dir "${BITSTREAM_DIR}" \
    --analysis_dir "${ANALYSIS_DIR}" \
    2>&1 | tee ${LOG_FILE}
