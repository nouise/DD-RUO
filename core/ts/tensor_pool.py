import torch
import random
from core.ts.tensor_data_func_v6 import TensorData, DPParams, TrainingHelper, _linear_schedule, TrainParam, pretrain_whole
from core.ts.core.quantizemodel import quantize_model_no_ref_v2
import concurrent.futures
import time
from torch import nn
from core.ts.training import loss_function


def all_to(param: DPParams, device):
    param = param.to(device)

def all_record(param: DPParams):
    return param.record()

def all_load_best(param: DPParams, device):
    param.load_best(device)

def all_load_best_param(param: DPParams, device):
    param.load_best_param(device)

def all_schedul_lr(param: DPParams):
    param.lr_schedule.step()


def run_model(model: TensorData, qp: TrainParam, param_device, target_device):
    model.to_run()
    y, noise, rt = model.mimic_forward(qp)
    npixels = y.shape[0] * y.shape[-1] * y.shape[-2]
    rt = rt.sum().item() / npixels
    y = y.to(target_device)
    noise = noise.to(target_device)
    model.to(param_device)
    return y, noise, rt


def run_model_test(model: TensorData, param_device, target_device):
    model.to_run()
    try:
        y, rt = model.forward_for_test()
    except Exception as e:
        print(f"run_error: run_model_test: {e}")
        raise
    npixels = y.shape[0] * y.shape[-1] * y.shape[-2]
    rt = torch.sum(rt).item() / npixels
    y = y.to(target_device)
    model.to(param_device)
    return y, rt


def run_quantize_net(model: TensorData, param, mse_err, param_device):
    # quantize_model_no_ref_v2 returns (best_loss, result_distribution, quant_param);
    # the main pipeline only needs bpp + components, so quant_param is dropped here.
    # entropy_codec/encode_v2.py uses the same call and keeps quant_param for the header.
    bpp, components, _quant_param = quantize_model_no_ref_v2(model, param, mse_err=mse_err)
    model.to(param_device)
    return bpp, components


def run_model_backward(model: TensorData, qp: TrainParam, data_grad, noise,
                       target_device, lr_scale, ldb_scale, epoch, cnt_param=0):
    running_device = model.device
    noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    model.empty_grad()
    y, rt = model.forward_per_sample(qp, noise)
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    npixels = data_grad.shape[0] * data_grad.shape[-1] * data_grad.shape[-2]
    bpp = drt.sum() / npixels
    rt_loss = bpp * qp.ldb
    rt_loss.backward()
    if epoch % 500 == 0 and cnt_param == 0:
        print(f"sum of grad:{torch.sum(torch.abs(data_grad))},{torch.sum(torch.abs(drt.grad))},"
              f"scale factor are:{lr_scale},{ldb_scale}")
        if model.dp.lr_schedule:
            print(f"step current lr:{model.dp.lr_schedule.state_dict()['_last_lr'][0]}")
    try:
        torch.autograd.backward([y, rt], [data_grad * lr_scale, drt.grad * ldb_scale])
        if epoch % 50 == 0 and cnt_param == 0:
            model.step(True)
        else:
            model.step(False)
    except Exception as e:
        print(f"An error occurred: {e}")
    model.to(target_device)
    return True


def run_warmup(model, ref, loss_fn, lmdb, max_iter, warmup_only):
    running_device = model.device
    ref = ref.to(running_device)
    param = pretrain_whole(model, ref, loss_fn, lmdb, max_iter, warmup_only)
    return param, ref


class ThreadPoolManager:
    def __init__(self, max_workers=3):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        print(f"Thread pool created with {max_workers} workers.")

    def clear_futures(self):
        self.futures = {}

    def __del__(self):
        self.shutdown()

    def submit_task(self, thread_key, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        self.futures[future] = thread_key
        return future

    def all_tasks_done(self):
        all_done = all(future.done() for future in self.futures.keys())
        while not all_done:
            time.sleep(0.001)
            all_done = all(future.done() for future in self.futures.keys())

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)
        print("Thread pool has been shut down.")


class TensorPool:

    def __init__(self, nclass, slice_size, sample_per_class: list[int],
                 cuda_devices_idxs: list[int], nthread=4,
                 ldb=0.001, img_size=(512, 512), max_iter=50000, channel=3,
                 lr=0.01, layers_v="v5", arm=32, dim=4,
                 freeze_modules=None) -> None:
        print(f"welcome to tensor pool, lr:{lr}, ldb:{ldb}, channel:{channel}, "
              f"img_size:{img_size}, cuda_devices_idxs:{cuda_devices_idxs}")
        if freeze_modules:
            print(f"Freeze modules: {freeze_modules}")
        self.freeze_modules = freeze_modules if freeze_modules else []
        self.nclass = nclass
        self.max_iter = max_iter
        self.slice_size = slice_size
        self.slice_per_class = [(ln + slice_size - 1) // slice_size for ln in sample_per_class]
        self.cuda_devices_idx = cuda_devices_idxs
        self.sample_per_class = sample_per_class
        self.nsample = sum(self.sample_per_class)
        self.slice_pool = {
            f'{cidx}_{sidx}': {'param': DPParams()}
            for cidx in range(nclass)
            for sidx in range(self.slice_per_class[cidx])
        }
        self.key_list = [
            f'{cidx}_{sidx}'
            for cidx in range(nclass)
            for sidx in range(self.slice_per_class[cidx])
        ]
        map_func = lambda sidx, cidx: (
            slice_size if sidx < self.slice_per_class[cidx] - 1
            else self.sample_per_class[cidx] % slice_size
        )
        self.slice_nums = [
            map_func(sidx, cidx)
            for cidx in range(nclass)
            for sidx in range(self.slice_per_class[cidx])
        ]
        self.label = torch.Tensor([
            k for k, nsample in enumerate(sample_per_class)
            for _ in range(nsample)
        ]).type(torch.long)
        self.target_device = f'cuda:{cuda_devices_idxs[0]}'
        self.nthread = nthread
        self.worker_pool = {
            tidx: TensorData(
                image_size=img_size, channel=channel,
                device=f'cuda:{cuda_devices_idxs[tidx % len(cuda_devices_idxs)]}',
                version=layers_v, arm=arm, dim=dim)
            for tidx in range(sum(self.slice_per_class))
        }
        self.helper = TrainingHelper(ldb, max_iter, lr)
        self.executor = ThreadPoolManager(nthread)
        self.ldb = ldb
        self.bidx = 0
        self.epoch = 0
        self.testing = False
        self.data_pool = None
        self.testing_data_redeay = False
        self.gain = len(self.key_list)

    def test(self):
        self.testing = True

    def train(self):
        self.testing = False
        self.testing_data_redeay = False

    def free_model(self):
        self.pool_idx = -1

    def get_model(self):
        self.pool_idx += 1
        return self.worker_pool[self.pool_idx]

    def record(self):
        for pk in self.slice_pool.keys():
            all_record(self.slice_pool[pk]['param'])

    def load_best(self):
        if not self.helper.dop.phase.schedule_lr:
            return
        if not self.helper.exist_best:
            return
        for pk in self.slice_pool.keys():
            all_load_best(self.slice_pool[pk]['param'], 'cpu')

    def load_best_param(self):
        for pk in self.slice_pool.keys():
            all_load_best_param(self.slice_pool[pk]['param'], 'cpu')

    def schedule_lr(self):
        for pk in self.slice_pool.keys():
            all_schedul_lr(self.slice_pool[pk]['param'])

    def set_training_phase(self, phase, freeze_latent=None):
        self.helper.set_training_phase(
            phase,
            freeze_latent=freeze_latent,
            freeze_modules=self.freeze_modules
        )
        self.helper.qp.start()

    def init_solvers(self):
        for pk in self.slice_pool.keys():
            self.helper.init_solver(self.slice_pool[pk]['param'],
                                    freeze_modules=self.freeze_modules)

    def get_data(self):
        with torch.no_grad():
            if self.testing:
                self.forward_test()
            else:
                self.forward_data()
            data = [self.data_pool[pk] for pk in self.key_list]
            rt = [self.rate_pool[pk] for pk in self.key_list]
        self.data = torch.nn.Parameter(torch.cat(data, dim=0), requires_grad=True)
        return self.data, self.label, sum(rt) / len(rt)

    @torch.no_grad()
    def fill_data_diff(self):
        grads = torch.split(self.data.grad, self.slice_nums)
        for id, pk in enumerate(self.key_list):
            self.data_pool[pk].grad = grads[id]

    def init_from_data(self, ref_dict):
        """ref_dict: {class_idx: tensor of reference images for that class}"""
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in ref_dict:
                nslice = self.slice_per_class[pk]
                for sidx in range(nslice):
                    ref = ref_dict[pk][sidx * self.slice_size:sidx * self.slice_size + self.slice_size]
                    current_model = self.get_model()
                    skey = f'{pk}_{sidx}'
                    self.executor.submit_task(skey, run_warmup, current_model, ref,
                                              loss_function, self.ldb, 5000, True)
            self.executor.all_tasks_done()
            self.free_model()
            for future in self.executor.futures.keys():
                skey = self.executor.futures[future]
                param, ref = future.result()
                self.slice_pool[skey]['param'].set_params(param.get_params(), 'cpu')
        except Exception as e:
            print(f'init_from_data fails: {e}')

    def save_slice_pool(self, pool_path, load_best=False):
        if load_best:
            self.load_best_param()
        for pk in self.key_list:
            self.slice_pool[pk]['noise'] = None
        torch.save(self.slice_pool, pool_path)

    def load_slice_pool(self, pool_path):
        self.slice_pool = torch.load(pool_path, map_location='cpu', weights_only=False)
        for skey in self.key_list:
            self.slice_pool[skey]['param'].load_reset()

    def validate(self, cnt):
        if self.helper.validate(cnt):
            print('-' * 20 + f'validate epoch {cnt}' + '-' * 20)
            self.test()
            self.schedule_lr()
            return True
        return False

    def forward_test(self):
        if self.testing_data_redeay:
            return
        self.rate_pool = {}
        self.data_pool = {}
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk, run_model_test, current_model,
                                          current_model.device, self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                y, rt = future.result()
                pk = self.executor.futures[future]
                self.data_pool[pk] = nn.Parameter(y, requires_grad=False)
                self.rate_pool[pk] = rt
        except Exception as e:
            print(f'forward_test fails: {e}')
        self.testing_data_redeay = True

    def quantize_net(self, mse_threshold):
        components_list = []
        bpp_list = []
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                self.executor.submit_task(
                    pk, run_quantize_net, current_model,
                    self.slice_pool[pk]['param'], mse_threshold,
                    current_model.device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                bpp, components = future.result()
                pk = self.executor.futures[future]
                formatted_output = ", ".join(
                    f"{key}={value:.6f}, ratio={value / bpp * 100:.2f}%"
                    for key, value in components.items()
                )
                print(f"pk={pk}: {formatted_output}")
                components_list.append(components)
                bpp_list.append(bpp)
            avg_bpp = sum(bpp_list) / len(bpp_list)
            avg_components = {
                key: sum(d[key] for d in components_list) / len(components_list)
                for key in components_list[0]
            }
            return avg_bpp, avg_components
        except Exception as e:
            print(f'quantize_net fails: {e}')

    def forward_data(self):
        self.rate_pool = {}
        self.data_pool = {}
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk, run_model, current_model,
                                          self.helper.qp, current_model.device,
                                          self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y, noise, rt = future.result()
                self.data_pool[pk] = nn.Parameter(y, requires_grad=True)
                self.data_pool[pk].grad = torch.zeros_like(self.data_pool[pk].data)
                self.slice_pool[pk]['noise'] = noise
                self.rate_pool[pk] = rt
        except Exception as e:
            print(f'forward_data fails: {e}')

    def backward(self, lr, ldb):
        self.free_model()
        try:
            self.executor.clear_futures()
            cnt = 0
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(
                    pk, run_model_backward, current_model, self.helper.qp,
                    self.data_pool[pk].grad, self.slice_pool[pk]['noise'],
                    current_model.device, lr, ldb, self.epoch, cnt)
                cnt += 1
            self.executor.all_tasks_done()
            self.epoch += 1
        except Exception as e:
            print(f'backward fails: {e}')
