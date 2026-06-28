#!/bin/bash
# Stage 1: TM (Trajectory Matching) distillation training with TensorPool
# ============================================================================
# Parameters split into TWO groups:
#   (A) TensorPool / joint-optimization params -- SHARED by TM / DM / DC.
#       Control the rate-utility trade-off of the compressed synthetic data.
#   (B) TM-specific distillation params (see bottom of this header).
#
# ---- beta: rate-distortion coefficient (warmup / init phase) ---------------
#   Paper beta is the RECIPROCAL of the warmup `ldb`:  beta = 1 / ldb_warmup.
#   (Code multiplies the coeff on the distortion/MSE term; the paper puts it
#    on the rate term -> reciprocal.) Warmup is auto-triggered by
#   init_from_data when pool_init="init" & no pool_init.pt exists: 5000 iters,
#   C3 'c3x' preset, start_lr=0.01, using args.ldb as the warmup ldb.
#   Paper TM/ImageNet:  beta = 10   <=>  warmup ldb = 0.1
#
# ---- lambda: rate-utility coefficient (joint-optimization phase) -----------
#   Effective paper lambda combines joint `ldb` with the two gradient
#   amplifiers (lr_it = distillation grad scale, ldb_it = rate grad scale):
#                 lambda = lr_it / (ldb_it * ldb)
#   Two-stage schedule: stage 1 LARGER lambda (prioritize distillation) ->
#   stage 2 SMALLER lambda (tighten bit-rate) by raising ldb_it.
#   Paper TM/ipc102  lambda = {1e3, 6.6e1}:
#     stage 1:  ldb_it=10   -> 1000/(10 *0.1) = 1e3
#     stage 2:  ldb_it=150  -> 1000/(150*0.1) = 6.6e1
#   (TM/ipc8,15: stage 2 ldb_it=120 -> 8.3e1, paper 8.5e1)
# ============================================================================

cd "$(dirname "$0")/.."

cuda_id=0,1,2,3
dst="ImageNet"
subset="imagenette"
net="ConvNetD5"                  # paper backbone: ConvNetD5 (Width=32, Depth=4 via arm/dim)
ipc=102                          # images-per-class (spc); paper ImageNet uses 102
sh_file="run_pool_tm.sh"
eval_mode="S"
data_path="/path/to/imagenet"

buffer_path="/path/to/buffers"   # (B) expert trajectory buffer dir (stage 0 output)

# ---- (A) TensorPool / joint-optimization params (shared TM/DM/DC) ----------
num_eval=5
Iteration=8000                   # joint-opt iterations (paper TM budget ~15000)
batch_syn=80                     # synthetic minibatch; 0 = full set (paper spc102 -> 80)
ldb=0.1                          # joint-opt rate coeff; enters lambda AND sets warmup beta=1/0.1=10
lr_img=0.001                     # Adam lr for latent+synthesis net, joint phase (paper 1e-3)
arm=32                           # synthesis/ARM width (paper Width=32)
dim=4                            # latent channel dim (paper Depth=4)
layers_v="v5"                    # synthesis network variant (paper TM: v5-240)
save_path="./results/tm/${dst}/${subset}/${ipc}"
pool_init="init"                 # "init" -> auto warmup (beta=1/ldb); or a path to pool_init.pt

mkdir -p ${save_path}
TAG="layers=${layers_v}_syn40_arm=${arm}_dim=${dim}_stage1_origin"
zca=False
lr_it=1000                       # (A) distillation-gradient amplifier (numerator of lambda)
res=128
ldb_it=10                        # (A) rate-gradient amplifier. stage1=10 (lambda=1e3); stage2=150 (lambda=6.6e1)
FLAG="${dst}_${subset}_${ipc}ipc_${net}_TM_pool_1_${Iteration}_${ldb}_${lr_img}_${lr_it}_${ldb_it}_${res}_zca_${zca}_#${TAG}"

# ---- (B) TM-specific distillation params -----------------------------------
# Trajectory-matching knobs (syn_steps, expert_epochs, max_start_epoch, lr_lr,
# lr_teacher) are NOT set here: TM/hyper_params.py load_default() auto-fills
# the paper ImageNet values -> syn_steps=40, expert_epochs=2,
# max_start_epoch=20, lr_lr=1e-5, lr_teacher=1e-2. Only buffer_path is needed.
# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${save_path}/${FLAG}_${timestamp}_${batch_syn}.log"
echo ${log_file}

export CUDA_VISIBLE_DEVICES=${cuda_id}
nohup python -u pool_tm.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} --pool_path ${pool_init} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--layers_v $layers_v \
--arm $arm \
--dim $dim \
--ldb ${ldb} --lr_img ${lr_img} --lr_it ${lr_it} --ldb_it ${ldb_it} \
--zca ${zca} \
--FLAG ${FLAG} > ${log_file} 2>&1 &
echo "Log file: ${log_file}"
echo "Process ID: $!"
