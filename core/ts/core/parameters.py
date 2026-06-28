from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from torch import nn, Tensor
import torch
from torch.nn.utils import clip_grad_norm_
import time
@dataclass
class CoolChicEncoderParameter:
    layers_synthesis: List[str] = field(default_factory=lambda: ['40-1-linear-relu','3-1-linear-none','3-3-residual-relu','3-3-residual-none'])
    n_ft_per_res: List[int] = field(default_factory=lambda:[1,1,1,1,1,1])
    dim_arm: int = 16#original_arm=8,best performance 32
    n_hidden_layers_arm: int = 4#从2调整为4
    upsampling_kernel_size: int = 8
    static_upsampling_kernel: bool = False
    encoder_gain: int = 16
    latent_n_grids: int = field(init=False)
    img_size: Optional[Tuple[int, int]] = field(init=False, default=None)
    device:torch.DeviceObjType = 'cuda:0'
    bitdepth:int = 16
    batch_size: int = 1

    def __post_init__(self):
        self.latent_n_grids = len(self.n_ft_per_res)

    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        self.img_size = img_size
        
    def set_deivce(self,device):
        self.device = device

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80
        s = "CoolChicEncoderParameter value:\n"
        s += "-------------------------------\n"
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s

def recursive_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().clone().detach()
    elif isinstance(data, dict):
        return {key: recursive_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_cpu(item) for item in data)
    else:
        return data
        
def recursive_to_device(data,device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: recursive_to_device(value,device) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(item,device) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_device(item,device) for item in data)
    else:
        return data
def analyze_gradient(param_grad):
    """分析梯度的分布情况"""
    flat_grad = param_grad.flatten()
    return {
        'max': flat_grad.abs().max().item(),
        'min': flat_grad.abs().min().item(),
        'mean': flat_grad.abs().mean().item(),
        'std': flat_grad.std().item(),
        'zeros': (flat_grad == 0).sum().item() / flat_grad.numel() * 100  # 零值百分比
    }

class MParams(nn.Module):
    
    def __init__(self)->None:
        self.iter = 0
        self.solver = None
        self.pool = {}
        self.all_parameters = None
        self.lr_schedule = None
        self.need_opt = False
        self.current_device = None        

    def load_reset(self):
        self.current_device = 'cpu'
    def print_lr(self):
        output_str = ""
        for param_group in self.solver.param_groups:
            output_str += f"Current learning rate: {param_group['lr']}\n"
        if self.lr_schedule is not None:
            output_str += f"lr_schedule: {self.lr_schedule.get_last_lr()[0]}\n"
        return output_str
    def init_solver(self,needs_opt:list[str],lr,max_itr,freq_valid,schedule=False,scale=1.0,freeze_modules=None):
        self.need_opt = False
        self.best_param = None
        self.best_opt = None
        self.all_parameters =  []
        parameters_to_optimize = []
        
        # 保存 freeze_modules 列表
        self.freeze_modules = freeze_modules if freeze_modules else []
        
        # 添加参数信息输出
        print("\n" + "="*50)
        print("Parameter Information:")
        if self.freeze_modules:
            print(f"Frozen modules: {self.freeze_modules}")
        total_params = 0
        trainable_params = 0
        
        # 添加参数控制字典，并根据 freeze_modules 设置
        self.grad_enabled = {
            "ap": True,
            "up": True,
            "sp": True,
            "grids": True
        }
        # 冻结指定模块：设置为 False，使其不加入优化器
        for module_name in self.freeze_modules:
            if module_name in self.grad_enabled:
                self.grad_enabled[module_name] = False
        for op in needs_opt:
            print(f"\nModule: {op}")
            print("-"*30)
            if self.grad_enabled.get(op, True):
                for i, param in enumerate(self.pool[op]):
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                    param.requires_grad = True
                    print(f"Parameter {i}:")
                    print(f"  Shape: {param.shape}")
                    print(f"  Requires grad: {param.requires_grad}")
                parameters_to_optimize += [*self.pool[op]]
            else:
                for i, param in enumerate(self.pool[op]):
                    total_params += param.numel()
                    param.requires_grad = False
                    print(f"Parameter {i}:")
                    print(f"  Shape: {param.shape}")
                    print(f"  Requires grad: {param.requires_grad}")
        
        print("\nSummary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*50 + "\n")
        
        if len(parameters_to_optimize) == 0: return
        print(f"init_solver:need_opt:{needs_opt},max_itr:{max_itr},freq_valid:{freq_valid},scale:{scale},lr:{lr},whether need schedule:{schedule}")
        lr_scale=lr*scale
        print(f"lr_scale:{lr_scale}")
        self.solver = torch.optim.Adam(parameters_to_optimize, lr=lr_scale)
        print(f"after init solver:{self.print_lr()}")
        if schedule:
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.solver, T_max= max_itr / freq_valid,
                #eta_min=0.00001, 考虑是否过于小
                eta_min=0.0005,
                last_epoch=-1,
            )
        else:
            self.lr_schedule = None
        for pk in self.pool.keys(): self.all_parameters += [*self.pool[pk]]
        self.need_opt = True
        
    def empty_grad(self):
        for param in self.all_parameters:  param.grad = None
    
    def record(self):
        self.best_opt = self.get_solver()
        self.best_param = self.get_params()

    
    def load_best(self, device):
        self.set_params(self.best_param,device)
        self.set_solver(self.best_opt,device)
        current_lr = self.lr_schedule.state_dict()["_last_lr"][0]
        for g in self.solver.param_groups: g["lr"] = current_lr
        
    def load_best_param(self,device):
        if not self.best_param is None:  self.set_params(self.best_param,device)
        self.best_param = None
        self.best_opt = None
        self.solver = None
        self.all_parameters = None
        
    def step(self,whether_print=False):
       # print('step',self.need_opt)
        if not self.need_opt: return
        #change the max norm to 1
        #clip_grad_norm_(self.all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)
        #clip_grad_norm_(self.all_parameters, 1, norm_type=2.0, error_if_nonfinite=False)
        def print_gradient_info(title):
            print(f"\n=== {title} ===")
            total_norm_sq = 0
            grad_stats = []
            
           # 第一遍循环：收集信息
            for param in self.all_parameters:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()  # L2范数
                    total_norm_sq += param_norm ** 2
                    grad_stats.append({
                        'shape': param.shape,
                        'norm': param_norm,
                        'analysis': analyze_gradient(param.grad.data)
                    })
            
            total_norm = total_norm_sq ** 0.5

            # 打印总体信息
            print(f"Total gradient norm: {total_norm:.6f}")
            print("\nDetailed gradient distribution:")
            print("{:<40} {:<10} {:<12} {:<10} {:<10} {:<10} {:<10}".format('Shape', 'Norm', 'Contribution', 'Max', 'Mean', 'Std', 'Zeros %'))
            print("-" * 82)

            # 第二遍循环：打印详细信息
            for i,stat in enumerate(grad_stats):
                print(f"the {i} th component:")
                contribution = (stat['norm'] ** 2 / total_norm_sq) * 100
                analysis = stat['analysis']
                print("{:>40} {:>10.6f} {:>12.2f}% {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(
                    str(stat['shape']), 
                    stat['norm'], 
                    contribution, 
                    analysis['max'], 
                    analysis['mean'], 
                    analysis['std'], 
                    analysis['zeros']
                ))
        
        # 打印裁剪前的梯度信息
        if whether_print:
            print_gradient_info("Before Gradient Clipping")
        # 执行梯度裁剪
        clip_grad_norm_(self.all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)        
        # 打印裁剪后的梯度信息
        if whether_print:
            print_gradient_info("After Gradient Clipping")
            print("get_params:",self.get_params_shape())
        self.solver.step()
        
        
    def get_params(self):
        snap = {}
        for k,v in self.pool.items():
            snap[k] = OrderedDict({tk:tv.cpu().clone().detach() for tk,tv in v.state_dict().items()})
        return snap
    #add new function
    def get_params_shape(self):
        shapes={}
        for k,v in self.pool.items():
            shapes[k]=OrderedDict({tk:tv.shape for tk,tv in v.state_dict().items()})
        return shapes
    def set_params(self,params_snap,device):
        self.current_device = device
        for k,v in params_snap.items():
            if self.pool[k] is None:
                self.pool[k] = nn.ParameterList([nn.Parameter(v[f'{i}'],requires_grad=True) for i in range(len(v))])
            else:
                self.pool[k].load_state_dict(v)
            self.pool[k] = self.pool[k].to(device)
                 
    def get_solver(self):
        if self.solver is None: return None
        return recursive_to_cpu(self.solver.state_dict())
    
    def set_solver(self,solver_snap,device):
        if not self.need_opt: return
        self.solver.load_state_dict(recursive_to_device(solver_snap,device))
        
    def to(self,device):
        #print('here',device,self.current_device)
        if device == self.current_device: return
        self.current_device = device
        for k,v in self.pool.items():
            self.pool[k] = v.to(device)
        if not self.solver is None:  self.solver.load_state_dict(recursive_to_device(self.solver.state_dict(),device))
        