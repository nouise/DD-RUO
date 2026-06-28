#!/bin/bash
# Stage 1: DM (Distribution Matching) distillation training with TensorPool
# ============================================================================
# Parameters split into TWO groups: (A) TensorPool / joint-opt params shared
# by TM/DM/DC, and (B) DM-specific distillation params (bottom of header).
#
# ---- beta: rate-distortion coefficient (warmup / init phase) ---------------
#   beta = 1 / ldb_warmup  (reciprocal; coeff sits on distortion in code, on
#   rate in paper). Paper DM/ImageNet:  beta = 1e6  <=>  warmup ldb = 1e-6.
#   IMPORTANT: the joint-opt `ldb` below is 5 (a DIFFERENT quantity, used in
#   lambda). To reproduce beta=1e6 you must run the warmup SEPARATELY with
#   ldb=1e-6 to produce pool_init.pt, then point pool_init at that file.
#   (Setting pool_init="init" here would auto-warmup with ldb=5 -> beta=0.2,
#    which is NOT the paper setting.) DM/DC share the same warmup pool_init.pt.
#
# ---- lambda: rate-utility coefficient (joint-optimization phase) -----------
#                 lambda = lr_it / (ldb_it * ldb)
#   Two-stage schedule. Paper DM lambda = {2e1, 6.7e-1}:
#     stage 1:  ldb_it=10   -> 1000/(10 *5) = 2e1   (imagenette used ldb_it=50)
#     stage 2:  ldb_it=300  -> 1000/(300*5) = 6.7e-1
# ============================================================================

cd "$(dirname "$0")/.."

cuda_id=0
dst="ImageNet"
subset="imagenette"
net="ConvNetD5"
ipc=96                           # paper DM/GM ImageNet uses spc=96
sh_file="run_pool_dm.sh"
eval_mode="S"
data_path="/path/to/imagenet"

# ---- (A) TensorPool / joint-optimization params (shared TM/DM/DC) ----------
num_eval=5
Iteration=10000                  # joint-opt iterations (paper DM budget ~20000)
batch_syn=0                      # 0 = use full synthetic set (paper DM batch=960)
ldb=5                            # joint-opt rate coeff; enters lambda (NOT the warmup beta)
lr_img=0.001                     # Adam lr for latent+synthesis net, joint phase (paper 1e-3)
arm=32                           # synthesis/ARM width (paper Width=32)
dim=4                            # latent channel dim (paper Depth=4)
layers_v="v6"                    # synthesis network variant
save_path="./results/dm/${dst}/${subset}/${ipc}"
pool_init="init"                 # see beta note above: use a pool_init.pt warmed up with ldb=1e-6

TAG="layers=${layers_v}_arm=${arm}_dim=${dim}"
zca=False
lr_it=1000                       # (A) distillation-gradient amplifier (numerator of lambda)
res=128
ldb_it=300                       # (A) rate-gradient amplifier. stage1=10/50 (lambda~2e1); stage2=300 (lambda=6.7e-1)
FLAG="${dst}_${subset}_${ipc}ipc_${net}_DM_pool_1_${Iteration}_${ldb}_${lr_img}_${lr_it}_${ldb_it}_${res}_zca_${zca}_#${TAG}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${save_path}/${FLAG}/${timestamp}_${batch_syn}.log"
echo ${log_file}
mkdir -p ${save_path}/${FLAG}/

# ---- (B) DM-specific distillation params (argparse defaults, paper-aligned) -
#   --lr_net=0.01 (network lr for distribution matching)
#   --batch_real=256, --dis_metric=ours
#   These use pool_dm.py defaults; override on the command line if needed.

export CUDA_VISIBLE_DEVICES=${cuda_id}
nohup python -u pool_dm.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
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
