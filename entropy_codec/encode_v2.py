"""
encode.py — 编码端

读 pool.pt → 取一个 slice → quantize_net → 提取 sent_params →
熵编码 latent（ARM + constriction 流式 CABAC）+ 编码网络权重 → 写到 OUT_DIR/

输入：
    POOL_PATH    pool .pt 文件
    SLICE_INDEX  取第几个 slice（默认 0）

输出（OUT_DIR）：
    header.bin                       纯二进制元信息（含 mask_size, ARM 配置等）
    latent_l{0..n_grids-1}.bin       每个 grid level 一个码流（所有 batch 张图按顺序拼成单一流）
    nn_arm_weight.bin / nn_arm_bias.bin
    nn_upsampling_weight.bin / nn_upsampling_bias.bin
    nn_synthesis_weight.bin / nn_synthesis_bias.bin
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchac
import constriction

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

# Importing core registers the `ts` -> `core.ts` backward-compat aliases in
# sys.modules, so legacy `from ts.xxx` imports and pre-reorg checkpoints keep
# working. We use the canonical `core.ts.xxx` paths below.
import core  # noqa: F401  (triggers the ts->core.ts shim in core/__init__.py)

from core.ts.tensor_data_func_v6 import TensorData
from core.ts.core.quantizemodel import quantize_model_no_ref_v2
from core.ts.core.misc import MAX_AC_MAX_VAL
from core.ts.core.arm_func import Arm, _get_neighbor, _get_non_zero_pixel_ctx_index
from codec_io import write_header, write_bin


# ---------- 配置（按需改） ----------
POOL_PATH = "/path/to/pool.pt"
OUT_DIR = os.path.join(THIS_DIR, "bitstream_out")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAYERS_V = "v5"
ARM_DIM = 32
N_HIDDEN_ARM = 4
MSE_ERR = 5e-7
SLICE_INDEX = 0

MASK_SIZE = 9            # ARM context mask 边长，必须与 pool 训练时的 mask_size 一致
LAPLACE_RANGE = 60       # constriction.QuantizedLaplace 的整数支撑区间 [-60, 60]
SCALE_FLOOR = 1e-6       # ARM 输出 scale 的下限


# ---------- torchac 工具 ----------
def laplace_cdf_table_shared(mu_scalar, scale_scalar, lo, hi, n):
    """所有 N 个位置共享 Laplace(mu, scale)，返回 (N, L+1) float32 CDF。"""
    syms = torch.arange(lo, hi + 1, dtype=torch.float64)
    L = syms.numel()
    mu = torch.tensor([[float(mu_scalar)]], dtype=torch.float64)
    scale = torch.tensor([[max(float(scale_scalar), 1e-6)]], dtype=torch.float64)
    syms_b = syms.view(1, L)

    def _cdf(x):
        return 0.5 + 0.5 * torch.sign(x - mu) * (1.0 - torch.exp(-(x - mu).abs() / scale))

    upper = _cdf(syms_b + 0.5)
    lower = _cdf(syms_b - 0.5)
    pmf = (upper - lower).clamp_min(1e-12)
    pmf = pmf / pmf.sum(dim=1, keepdim=True)
    cdf_inner = torch.cumsum(pmf, dim=1)
    cdf_inner = cdf_inner / cdf_inner[:, -1:]
    cdf = torch.cat([torch.zeros(1, 1, dtype=torch.float64), cdf_inner], dim=1)
    cdf = cdf.to(torch.float32).clamp(0.0, 1.0)
    cdf[:, 0] = 0.0
    cdf[:, -1] = 1.0
    return cdf.expand(n, -1).contiguous()


def encode_with_cdf(sent_int, cdf, out_path):
    sent_int = sent_int.to(torch.long).cpu()
    cdf = cdf.to(torch.float32).cpu().contiguous()
    L = cdf.shape[-1] - 1
    assert sent_int.min() >= 0 and sent_int.max() < L, (
        f"sent index out of [0,{L}): min={sent_int.min().item()}, max={sent_int.max().item()}"
    )
    sym_int16 = sent_int.to(torch.int16)
    byte_stream = torchac.encode_float_cdf(
        cdf.unsqueeze(0), sym_int16.unsqueeze(0),
        check_input_bounds=False, needs_normalization=True,
    )
    write_bin(out_path, byte_stream)
    return len(byte_stream)


def make_context_lookup(H, W, mask_size=9, arm_dim=32):
    """与 decode_v2 完全相同的 context lookup，保证编解码浮点路径一致"""
    center = (mask_size - 1) // 2
    mask_idx = _get_non_zero_pixel_ctx_index(arm_dim)
    dy = mask_idx // mask_size - center
    dx = mask_idx % mask_size - center
    n = H * W
    ys = torch.arange(n) // W
    xs = torch.arange(n) % W
    ny = ys[:, None] + dy[None, :]
    nx = xs[:, None] + dx[None, :]
    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    flat_idx = torch.where(valid, ny * W + nx, torch.tensor(-1, dtype=torch.long))
    return flat_idx


# ---------- 主流程 ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # [1] 加载 pool
    print(f"[1] loading {POOL_PATH}")
    slice_pool = torch.load(POOL_PATH, map_location="cpu", weights_only=False)
    keys = list(slice_pool.keys())
    pk = keys[SLICE_INDEX]
    print(f"    n_slices = {len(keys)}, slice key = {pk}")

    dp = slice_pool[pk]["param"]
    dp.load_reset()
    grids = dp.pool["grids"]
    batch, _, H, W = grids[0].shape
    im_size = (H, W)
    channel = dp.pool["sp"][4].shape[0]
    print(f"    slice batch={batch}, im_size={im_size}, channel={channel}, "
          f"#latent_levels={len(grids)}")

    # [2] 量化网络
    print(f"\n[2] running quantize_model_no_ref_v2 (mse_err={MSE_ERR}) ...")
    op = TensorData(
        image_size=im_size, channel=channel, device=DEVICE,
        version=LAYERS_V, arm=ARM_DIM, dim=N_HIDDEN_ARM,
    )
    op.set_param(dp)
    op.to_run()
    t0 = time.time()
    best_bpp, distribution, quant_param = quantize_model_no_ref_v2(op, dp, mse_err=MSE_ERR)
    print(f"    done in {time.time()-t0:.1f}s, ARM-estimate bpp = {best_bpp:.6f}")

    # [3] 网络 sent_params
    print(f"\n[3] extracting network sent_params ...")
    mod_to_pool = {"arm": "ap", "upsampling": "up", "synthesis": "sp"}
    nn_sent, nn_shapes = {}, {}
    for mod_name, pool_key in mod_to_pool.items():
        qs = quant_param[mod_name]["best_q_step"]
        sent_w, sent_b = [], []
        layer_meta = []
        for p in dp.pool[pool_key]:
            is_w = len(p.shape) >= 2
            qstep = qs["weight"] if is_w else qs["bias"]
            sent = torch.round(p.detach().cpu() / qstep).to(torch.long)
            (sent_w if is_w else sent_b).append(sent.view(-1))
            layer_meta.append({"shape": tuple(p.shape), "is_weight": is_w, "numel": p.numel()})
        sw = torch.cat(sent_w) if sent_w else torch.tensor([], dtype=torch.long)
        sb = torch.cat(sent_b) if sent_b else torch.tensor([], dtype=torch.long)
        nn_sent[mod_name] = {"weight": sw, "bias": sb}
        nn_shapes[mod_name] = layer_meta
        print(f"    {mod_name:11s}: w n={sw.numel():6d}  b n={sb.numel():4d}  "
              f"q_step(w)={float(qs['weight']):.6f}  q_step(b)={float(qs['bias']):.6f}")

    # [4] 取 grids 各 level 的 sent_int（保留 (batch, 1, H, W) 维度）
    print(f"\n[4] computing latent sent (per-level, keep batch dim) ...")
    op.set_to_eval()
    encoder_gain = op.gen.encoder_gains
    with torch.no_grad():
        sent_per_level = []
        for g in dp.pool["grids"]:
            sent_l = torch.round(g.detach() * encoder_gain)
            # 边界保护：超出 LAPLACE_RANGE 直接断言（constriction 模型支撑区间是 ±60）
            if sent_l.abs().max().item() > LAPLACE_RANGE:
                raise ValueError(
                    f"latent sent exceeds LAPLACE_RANGE={LAPLACE_RANGE}: "
                    f"actual max abs = {sent_l.abs().max().item()}, level shape={tuple(g.shape)}"
                )
            sent_per_level.append(sent_l.cpu())  # (batch, 1, H, W)
    print(f"    {len(sent_per_level)} levels, batch={sent_per_level[0].shape[0]}")
    for lvl, s in enumerate(sent_per_level):
        print(f"      L{lvl} {tuple(s.shape)} sent range=[{int(s.min())},{int(s.max())}]")

    # 准备 ARM (用 dp 自带的 ap) — 编码端 ARM 在 CPU 上跑（与解码端一致）
    arm = Arm(ARM_DIM, N_HIDDEN_ARM)
    arm.initialize_parameters_map()
    arm.eval()
    # 把 ap 搬到 CPU（quantize 后 ap 可能在 cuda 上）
    ap = nn.ParameterList([
        nn.Parameter(p.detach().cpu(), requires_grad=False) for p in dp.pool["ap"]
    ])
    # [5] 编码 latent（逐像素单向量 ARM，与 decode_v2 浮点路径完全一致）
    print(f"\n[5] encoding latent per-level (B v2: per-pixel ARM) -> {OUT_DIR}/")
    file_records = []
    laplace_model = constriction.stream.model.QuantizedLaplace(-LAPLACE_RANGE, LAPLACE_RANGE)
    total_lat_bytes = 0
    for lvl, sent_layer in enumerate(sent_per_level):
        batch_l, _, H_l, W_l = sent_layer.shape
        n_per_img = H_l * W_l
        ctx_lookup = make_context_lookup(H_l, W_l, MASK_SIZE, ARM_DIM)
        encoder = constriction.stream.queue.RangeEncoder()
        for img_idx in range(batch_l):
            sent_2d = sent_layer[img_idx:img_idx+1].to(torch.float32)
            flat_s = sent_2d.view(-1)
            with torch.no_grad():
                for i in range(n_per_img):
                    idx = ctx_lookup[i]
                    cv = flat_s[idx.clamp(min=0)]
                    cv[idx < 0] = 0.0
                    mu_t, sc_t = arm(cv.unsqueeze(0), ap)[:2]
                    mu_v = mu_t[0].item()
                    sc_v = max(sc_t[0].item(), SCALE_FLOOR)
                    sym_i = int(flat_s[i].item())
                    encoder.encode(
                        np.array([sym_i], dtype=np.int32),
                        laplace_model,
                        np.array([mu_v], dtype=np.float32),
                        np.array([sc_v], dtype=np.float32),
                    )
        compressed = encoder.get_compressed()
        path_l = os.path.join(OUT_DIR, f"latent_l{lvl}.bin")
        write_bin(path_l, compressed.tobytes())
        nb = os.path.getsize(path_l)
        file_records.append((f"latent_l{lvl}.bin", nb))
        total_lat_bytes += nb
        print(f"    latent_l{lvl}.bin  shape=({batch_l},1,{H_l},{W_l})  "
              f"n_per_img={n_per_img}  total_syms={batch_l*n_per_img}  bytes={nb}")
    print(f"    latent total bytes = {total_lat_bytes}")

    # [6] 编码网络（每流单 Laplace(0, sigma_MLE)）
    print(f"\n[6] encoding network streams ...")
    nn_meta = {}
    for mod_name, parts in nn_sent.items():
        nn_meta[mod_name] = {}
        for kind in ("weight", "bias"):
            sent = parts[kind]
            n = sent.numel()
            fname = f"nn_{mod_name}_{kind}.bin"
            path = os.path.join(OUT_DIR, fname)
            if n == 0:
                write_bin(path, b"")
                file_records.append((fname, os.path.getsize(path)))
                nn_meta[mod_name][kind] = {"n": 0}
                print(f"    {fname:35s}empty")
                continue
            sigma = max(float(sent.abs().to(torch.float32).mean().item()), 0.5)
            lo = int(sent.min()) - 4
            hi = int(sent.max()) + 4
            cdf = laplace_cdf_table_shared(0.0, sigma, lo, hi, n)
            idx = (sent - lo).contiguous()
            nb = encode_with_cdf(idx, cdf, path)
            file_records.append((fname, nb))
            nn_meta[mod_name][kind] = {"n": n, "lo": lo, "hi": hi, "sigma": sigma}
            print(f"    {fname:35s}n={n:6d}  sigma={sigma:.3f}  bytes={nb}")

    # [7] header
    eg_val = float(encoder_gain) if not torch.is_tensor(encoder_gain) or encoder_gain.numel() == 1 \
             else float(encoder_gain.flatten()[0])
    header = {
        "channel": channel, "batch": batch, "im_size": im_size,
        "arm_dim": ARM_DIM, "n_hidden_arm": N_HIDDEN_ARM,
        "mask_size": MASK_SIZE,
        "layers_v": LAYERS_V,
        "encoder_gain": eg_val,
        "latent_shapes": [tuple(g.shape) for g in dp.pool["grids"]],
        "modules": [
            {
                "name": mn, "layers": nn_shapes[mn],
                "q_step_w": float(quant_param[mn]["best_q_step"]["weight"]),
                "q_step_b": float(quant_param[mn]["best_q_step"]["bias"]),
                "weight": nn_meta[mn]["weight"], "bias": nn_meta[mn]["bias"],
            }
            for mn in ("arm", "upsampling", "synthesis")
        ],
    }
    header_path = os.path.join(OUT_DIR, "header.bin")
    header_size = write_header(header_path, header)
    file_records.append(("header.bin", header_size))
    print(f"    header.bin                         bytes={header_size}")

    # [8] summary
    total = sum(b for _, b in file_records)
    npix = batch * im_size[0] * im_size[1]
    print(f"\n[8] DONE")
    print(f"    output dir         = {OUT_DIR}")
    print(f"    bitstream total    = {total:,} bytes ({total/1024:.2f} KB)")
    print(f"    bpp (actual)       = {total*8/npix:.6f}")
    print(f"    bpp (ARM-estimate) = {best_bpp:.6f}")
    for name, nb in file_records:
        print(f"      {name:<35s}{nb:>8d} bytes ({nb/total*100:5.2f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="entropy_codec encode_v2: pool.pt -> bitstream (ARM streaming CABAC)")
    p.add_argument("--pool_path", type=str, default=POOL_PATH)
    p.add_argument("--slice_index", type=int, default=SLICE_INDEX)
    p.add_argument("--layers_v", type=str, default=LAYERS_V)
    p.add_argument("--arm_dim", type=int, default=ARM_DIM)
    p.add_argument("--n_hidden_arm", type=int, default=N_HIDDEN_ARM)
    p.add_argument("--mse_err", type=float, default=MSE_ERR)
    p.add_argument("--mask_size", type=int, default=MASK_SIZE)
    p.add_argument("--laplace_range", type=int, default=LAPLACE_RANGE)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = p.parse_args()
    POOL_PATH = args.pool_path
    SLICE_INDEX = args.slice_index
    LAYERS_V = args.layers_v
    ARM_DIM = args.arm_dim
    N_HIDDEN_ARM = args.n_hidden_arm
    MSE_ERR = args.mse_err
    MASK_SIZE = args.mask_size
    LAPLACE_RANGE = args.laplace_range
    OUT_DIR = args.out_dir
    main()
