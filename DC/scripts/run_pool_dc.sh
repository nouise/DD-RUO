#!/bin/bash
# Stage 1/2: DC (Gradient Matching; called "GM" in the paper) with TensorPool
# ============================================================================
# Parameters split into TWO groups: (A) TensorPool / joint-opt params shared
# by TM/DM/DC, and (B) DC-specific distillation params (bottom of header).
#
# ---- beta: rate-distortion coefficient (warmup / init phase) ---------------
#   beta = 1 / ldb_warmup. Paper GM/ImageNet: beta = 1e6 <=> warmup ldb = 1e-6.
#   The joint-opt `ldb` below is 5 (a DIFFERENT quantity used in lambda).
#   Reproduce beta=1e6 via a SEPARATE warmup run with ldb=1e-6 producing
#   pool_init.pt (DC reuses the DM warmup pool_init.pt), then point pool_init
#   at it for stage 1.
#
# ---- lambda: rate-utility coefficient (joint-optimization phase) -----------
#                 lambda = lr_it / (ldb_it * ldb)
#   Two-stage schedule. Paper GM lambda = {2.8e-2, 1.2e-2}:
#     stage 1:  ldb_it=700   -> 100/(700 *5) = 2.8e-2
#     stage 2:  ldb_it=1500  -> 100/(1500*5) = 1.3e-2
#   NOTE: this script is configured for STAGE 2 (loads a stage-1 checkpoint via
#   pool_init). For stage 1, set ldb_it=700 and pool_init to the warmup .pt.
# ============================================================================

cd "$(dirname "$0")/.."

cuda_id=0
dst="ImageNet"
subset="imagefruit"
net="ConvNetD5"
ipc=96                           # paper GM ImageNet uses spc=96
sh_file="run_pool_dc.sh"
eval_mode="S"
data_path="/path/to/imagenet"

# ---- (A) TensorPool / joint-optimization params (shared TM/DM/DC) ----------
num_eval=5
Iteration=500                    # joint-opt iterations (paper GM budget ~800)
batch_syn=0                      # 0 = use full synthetic set
ldb=5                            # joint-opt rate coeff; enters lambda (NOT the warmup beta)
lr_img=0.001                     # Adam lr for latent+synthesis net, joint phase (paper 1e-3)
arm=32                           # synthesis/ARM width (paper Width=32)
dim=4                            # latent channel dim (paper Depth=4)
layers_v="v6"                    # synthesis network variant
TAG="layers=${layers_v}_arm=${arm}_dim=${dim}_stage2_50_20"
zca=False
lr_it=100                        # (A) distillation-gradient amplifier (numerator of lambda)
res=128
ldb_it=1500                      # (A) rate-gradient amplifier. stage1=700 (lambda=2.8e-2); stage2=1500 (lambda=1.3e-2)
save_path="./results/dc/${dst}/${subset}/${ipc}"
# Stage-2 loads the stage-1 pool checkpoint. For stage 1, set this to the
# warmup pool_init.pt (ldb=1e-6 warmup) instead.
pool_init="/path/to/stage1/..._dc_pool.pt"

method="DC"                      # (B) DC = Gradient Matching (paper "GM")

FLAG="${dst}_${subset}_${ipc}ipc_${net}_${method}_DC_pool_1_${Iteration}_${ldb}_${lr_img}_${lr_it}_${ldb_it}_${res}_zca_${zca}_#${TAG}"
# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${save_path}/${FLAG}/${timestamp}_${batch_syn}.log"
echo ${log_file}
mkdir -p ${save_path}/${FLAG}/

# ---- (B) DC-specific distillation params (argparse defaults, paper-aligned) -
#   --method=DC, --lr_net=0.01, --batch_real=256, --init=real, --dis_metric=ours
#   outer_loop / inner_loop are auto-set by get_loops(ipc) in pool_dc.py.
#   These use pool_dc.py defaults; override on the command line if needed.

export CUDA_VISIBLE_DEVICES=${cuda_id}
nohup python -u pool_dc.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--method ${method} \
--ipc ${ipc} \
--layers_v $layers_v \
--arm $arm \
--dim $dim \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --pool_path ${pool_init} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--zca ${zca} \
--batch_syn ${batch_syn} \
--ldb ${ldb} --lr_img ${lr_img} --lr_it ${lr_it} --ldb_it ${ldb_it} \
--FLAG ${FLAG} > ${log_file} 2>&1 &
echo "Log file: ${log_file}"
echo "Process ID: $!"

