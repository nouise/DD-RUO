# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import itertools
import time
from typing import Optional, OrderedDict

import torch
from core.ts.core.misc import exp_golomb_nbins
from core.ts.core.misc import (
    MAX_AC_MAX_VAL,
    POSSIBLE_EXP_GOL_COUNT,
    POSSIBLE_Q_STEP,
    get_q_step_from_parameter_name,
)
from torch import Tensor


def _quantize_parameters(
    fp_param: torch.nn.ParameterList,
    q_step,
) -> torch.nn.ParameterList:

    q_param = torch.nn.ParameterList()
    param_int = {'weight':[],'bias':[]}
    for pid in range(len(fp_param)):
        is_weight = len(fp_param[pid].shape)>=2
        current_q_step = q_step['weight'] if is_weight else q_step['bias']
        sent_param = torch.round(fp_param[pid] / current_q_step)

        if sent_param.abs().max() > MAX_AC_MAX_VAL:
            return None,None
        q_param.append(sent_param * current_q_step) 
        if is_weight: param_int['weight'].append(sent_param.view(-1))
        else: param_int['bias'].append(sent_param.view(-1))
        

    return q_param, param_int

def cal_nparam(params:torch.nn.ParameterList):
    cur_total = 0
    for pid in range(len(params)):
        param = params[pid]
        cur_size = param.numel()
        cur_total += cur_size
    return cur_total*32

@torch.no_grad()
def quantize_model_img(op, param, loss_func) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'arm':param.pool['ap'],'upsampling':param.pool['up'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param = {'arm':None,'upsampling':None,'synthesis':None}
    y,bits = op.forward_for_test()
    print(bits_per_module)
    loss,mse,bpp = loss_func.cal(y,bits,0)
    print(loss.item(),mse.item(),bpp.item())
    for module_name, cur_module in sorted(module_to_quantize.items()):
        best_loss = 1e6
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        print(module_name)
        for q_step_w, q_step_b in itertools.product(all_q_step.get("weight"), all_q_step.get("bias")):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param,param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None:   continue

            param.pool[module_name] = q_param

            y,bits = op.forward_for_test()

            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in all_expgol_cnt.get(worb):
                    cur_rate = exp_golomb_nbins(v, count=expgol_cnt).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            loss,mse,bpp = loss_func.cal(y,bits,net_bits+bits_base)
            print(q_step_w, q_step_b,loss.item(),mse.item(),bpp.item(),net_bits,bits_base)

            # Store best quantization steps
            if loss < best_loss:
                best_loss = loss
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                print(bits_per_module)

        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
       
    print(bits_per_module)
    time_nn_quantization = time.time() - start_time

    return param
def compare_two_nets(current_param,q_param,module_name):
            # 新增：比较q_param和当前pool中的参数
            total_diff = 0.0
            for q_p, c_p in zip(q_param, current_param):
                    diff = torch.sum(torch.abs(q_p - c_p)).item()
                    total_diff += diff
                    print(f"shape:{q_p.shape},Parameter diff: {diff:.6f}")
                
            print(f"Total absolute difference for {module_name}: {total_diff:.6f}")
@torch.no_grad()
def quantize_model_no_ref(op, param,mse_err=0.00005) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'upsampling':param.pool['up'],'arm':param.pool['ap'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param = {'arm':None,'upsampling':None,'synthesis':None}
    name_translate={"arm":"ap","upsampling":"up","synthesis":"sp"}
    y,bits = op.forward_for_test()

    ref = y.clone().detach()
    npixels = ref.numel() / ref.shape[1]
    simple_latents=torch.sum(bits).item()/npixels
    print(f"before:{torch.sum(bits).item()/npixels},npixels:{npixels},bits_per_module:{bits_per_module}")
    final_mse=0.0
    before_bits_per_module=bits_per_module.copy()
    best_loss=1e6
    for module_name, cur_module in sorted(module_to_quantize.items()):
        #best_loss = Max_best_loss
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        best_q_param=param.pool[name_translate[module_name]]
        for q_step_w, q_step_b in itertools.product(all_q_step.get("weight"), all_q_step.get("bias")):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param, param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None: continue
            param.pool[name_translate[module_name]] = q_param
            op.set_param(param)
            y,bits = op.forward_for_test()
            
            mse = torch.mean((y-ref)**2)
            if mse > mse_err: continue
            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in all_expgol_cnt.get(worb):
                    cur_rate = exp_golomb_nbins(v, count=expgol_cnt).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            bpp = (net_bits+bits_base+torch.sum(bits).item())/npixels
            # Store best quantization steps
            if bpp < best_loss:
                best_loss = bpp
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                simple_latents=torch.sum(bits).item()/npixels
                final_mse=mse
                print(q_step_w, q_step_b,bpp,net_bits,bits_base)
                best_q_param=q_param
                if module_name=="synthesis":
                    print(fp_param[0][0],fp_param[0].shape)
                    print(q_param[0][0])
                    print(param_int['weight'][0][:6])
                    print(f"q_step_w,q_step_b:{q_step_w},{q_step_b}")
        param.pool[name_translate[module_name]]=best_q_param
        compare_two_nets(fp_param,param.pool[name_translate[module_name]],module_name)
        compare_two_nets(best_q_param,param.pool[name_translate[module_name]],f"{module_name}_best_q_param")
        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
    print(f"before_bpp{simple_latents},final bpp{best_loss};ratio:{(simple_latents/best_loss)*100}")   
    print("quant_param:",quant_param)
    print(f"before:{before_bits_per_module}")
    #print(f"after:{bits_per_module}")
    result_distribution={}
    for module_name, bits in bits_per_module.items():
        result_distribution[module_name]=bits/npixels
    result_distribution["grids"]=simple_latents
    print(f"after:{result_distribution}")

    print(f"param.pool.keys:{param.pool.keys()}")
    time_nn_quantization = time.time() - start_time
    op.set_param(param)
    y,bits = op.forward_for_test()
            
    mse = torch.mean((y-ref)**2)
    print(time_nn_quantization)
    return param,best_loss,simple_latents,time_nn_quantization,final_mse.item(),mse.item(),ref,y,result_distribution

@torch.no_grad()
def quantize_model_no_ref_v2(op, param,mse_err=0.00005) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'upsampling':param.pool['up'],'arm':param.pool['ap'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param = {'arm':None,'upsampling':None,'synthesis':None}
    name_translate={"arm":"ap","upsampling":"up","synthesis":"sp"}
    y,bits = op.forward_for_test()

    ref = y.clone().detach()
    npixels = ref.numel() / ref.shape[1]
    simple_latents=torch.sum(bits).item()/npixels
    final_mse=0.0
    before_bits_per_module=bits_per_module.copy()
    best_loss=1e6
    for module_name, cur_module in sorted(module_to_quantize.items()):
        #best_loss = Max_best_loss
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        best_q_param=param.pool[name_translate[module_name]]
        for q_step_w, q_step_b in itertools.product(all_q_step.get("weight"), all_q_step.get("bias")):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param, param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None: continue
            param.pool[name_translate[module_name]] = q_param
            op.set_param(param)
            y,bits = op.forward_for_test()
            
            mse = torch.mean((y-ref)**2)
            if mse > mse_err: continue
            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in all_expgol_cnt.get(worb):
                    cur_rate = exp_golomb_nbins(v, count=expgol_cnt).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            bpp = (net_bits+bits_base+torch.sum(bits).item())/npixels
            # Store best quantization steps
            if bpp < best_loss:
                best_loss = bpp
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                simple_latents=torch.sum(bits).item()/npixels
                final_mse=mse
                best_q_param=q_param
        param.pool[name_translate[module_name]]=best_q_param
        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
    result_distribution={}
    for module_name, bits in bits_per_module.items():
        result_distribution[module_name]=bits/npixels
    result_distribution["grids"]=simple_latents
    for op in ['ap','up','sp']:
        for i, param_single in enumerate(param.pool[op]):
                        param_single.requires_grad = True
    return best_loss,result_distribution,quant_param