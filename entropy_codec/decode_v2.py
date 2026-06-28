"""
decode_v2.py — 解码端

读码流目录 → 解码网络权重与 latent（ARM + constriction 流式 CABAC）→ 合成图像 → 写到 OUT_DIR/。
用法见 README.md / scripts/run_decode.sh。
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torchac
import torchvision.utils as vutils
import constriction
from tqdm import tqdm
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import core  # noqa: F401  (triggers the ts->core.ts shim in core/__init__.py)

from core.ts.tensor_data_func_v6 import TensorData, DPParams
from core.ts.core.arm_func import Arm, _get_neighbor, _get_non_zero_pixel_ctx_index
from codec_io import read_header, read_bin

# ---------- 配置 ----------
LAPLACE_RANGE = 60
SCALE_FLOOR = 1e-6
DECODE_N_IMAGES = None  # None = 全量 51 张，设数字则只解前 N 张调试用
BITSTREAM_DIR = os.path.join(THIS_DIR, "bitstream_out")
OUT_DIR = os.path.join(THIS_DIR, "decoded_out")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------- context 预计算 ----------
def make_context_lookup(H, W, mask_size=9, arm_dim=32):
    center = (mask_size - 1) // 2
    mask_idx = _get_non_zero_pixel_ctx_index(arm_dim)  # (arm_dim,)
    dy = mask_idx // mask_size - center
    dx = mask_idx % mask_size - center

    n = H * W
    ys = torch.arange(n) // W
    xs = torch.arange(n) % W
    ny = ys[:, None] + dy[None, :]  # (n, arm_dim)
    nx = xs[:, None] + dx[None, :]
    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    flat_idx = torch.where(valid, ny * W + nx, torch.tensor(-1, dtype=torch.long))
    return flat_idx  # (n, arm_dim)


def decode_latent_level_fast(raw_bytes, H, W, ctx_lookup, arm, ap, laplace_model, n_imgs):
    """优化版：precomputed context lookup + 单向量 ARM（需 encode_v2.py 配套）"""
    n_per_img = H * W
    compressed = np.frombuffer(raw_bytes, dtype=np.uint32).copy()
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    sent_layer = torch.zeros(n_imgs, 1, H, W)
    for img_idx in tqdm(range(n_imgs), desc=f"    ({H}×{W})", unit="img"):
        flat_dec = torch.zeros(n_per_img)
        with torch.no_grad():
            for i in range(n_per_img):
                idx = ctx_lookup[i]
                cv = flat_dec[idx.clamp(min=0)]
                cv[idx < 0] = 0.0
                mu_t, sc_t = arm(cv.unsqueeze(0), ap)[:2]
                mu_v, sc_v = mu_t[0].item(), max(sc_t[0].item(), SCALE_FLOOR)
                sym = decoder.decode(laplace_model, np.array([mu_v], dtype=np.float32), np.array([sc_v], dtype=np.float32))
                flat_dec[i] = float(sym[0])
        sent_layer[img_idx] = flat_dec.view(1, H, W)
    return sent_layer


# ---------- torchac CDF ----------
def laplace_cdf_table_shared(mu, scale, lo, hi, n):
    syms = torch.arange(lo, hi + 1, dtype=torch.float64)
    L = syms.numel()
    mu_t = torch.tensor([[float(mu)]], dtype=torch.float64)
    sc_t = torch.tensor([[max(float(scale), 1e-6)]], dtype=torch.float64)
    syms_b = syms.view(1, L)

    def _cdf(x):
        return 0.5 + 0.5 * torch.sign(x - mu_t) * (1.0 - torch.exp(-(x - mu_t).abs() / sc_t))

    upper = _cdf(syms_b + 0.5)
    lower = _cdf(syms_b - 0.5)
    pmf = (upper - lower).clamp_min(1e-12)
    pmf = pmf / pmf.sum(dim=1, keepdim=True)
    cdf_inner = torch.cumsum(pmf, dim=1)
    cdf_inner = cdf_inner / cdf_inner[:, -1:]
    cdf = torch.cat([torch.zeros(1, 1, dtype=torch.float64), cdf_inner], dim=1)
    cdf = cdf.to(torch.float32).clamp(0.0, 1.0)
    cdf[:, 0] = 0.0; cdf[:, -1] = 1.0
    return cdf.expand(n, -1).contiguous()


def decode_with_cdf(cdf, bs):
    cdf = cdf.to(torch.float32).cpu().contiguous()
    sym = torchac.decode_float_cdf(cdf.unsqueeze(0), bs, needs_normalization=True)
    return sym.squeeze(0).to(torch.long)


# ---------- 主流程 ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, "decoded_images")
    os.makedirs(img_dir, exist_ok=True)

    # [1] 读 header
    print(f"[1] reading header.bin ...")
    h = read_header(os.path.join(BITSTREAM_DIR, "header.bin"))
    H, W = h["im_size"]; ch = h["channel"]; batch = h["batch"]
    eg = h["encoder_gain"]; n_grids = len(h["latent_shapes"])
    mask_size = h["mask_size"]; arm_dim = h["arm_dim"]; n_hidden = h["n_hidden_arm"]
    print(f"    {H}×{W} ch={ch} batch={batch} layers={h['layers_v']} arm={arm_dim}×{n_hidden} mask={mask_size}")

    # [2] 文件清单
    expected = ["header.bin"] + [f"latent_l{i}.bin" for i in range(n_grids)] + [
        "nn_arm_weight.bin", "nn_arm_bias.bin",
        "nn_upsampling_weight.bin", "nn_upsampling_bias.bin",
        "nn_synthesis_weight.bin", "nn_synthesis_bias.bin",
    ]
    total = sum(os.path.getsize(os.path.join(BITSTREAM_DIR, n)) for n in expected)
    print(f"[2] bitstream: {total:,} bytes ({total/1024:.2f} KB)")

    # [3] 解码网络
    print(f"[3] decoding network weights ...")
    nn_dec = {}
    for mod_name in ("arm", "upsampling", "synthesis"):
        m = next(mm for mm in h["modules"] if mm["name"] == mod_name)
        nn_dec[mod_name] = {"meta": m}
        for kind in ("weight", "bias"):
            mk = m[kind]; n = mk["n"]
            if n == 0:
                nn_dec[mod_name][kind] = torch.tensor([], dtype=torch.long)
                continue
            cdf = laplace_cdf_table_shared(0.0, mk["sigma"], mk["lo"], mk["hi"], n)
            bs = read_bin(os.path.join(BITSTREAM_DIR, f"nn_{mod_name}_{kind}.bin"))
            nn_dec[mod_name][kind] = decode_with_cdf(cdf, bs) + mk["lo"]

    # 4a 重建 ARM 参数
    arm_meta = nn_dec["arm"]["meta"]
    qsw, qsb = arm_meta["q_step_w"], arm_meta["q_step_b"]
    ap_parts = []; iw = ib = 0
    sw_a, sb_a = nn_dec["arm"]["weight"], nn_dec["arm"]["bias"]
    for lm in arm_meta["layers"]:
        sh, n_l, is_w = lm["shape"], lm["numel"], lm["is_weight"]
        if is_w:
            ap_parts.append(sw_a[iw:iw+n_l].float() * qsw); iw += n_l
        else:
            ap_parts.append(sb_a[ib:ib+n_l].float() * qsb); ib += n_l
        ap_parts[-1] = ap_parts[-1].view(*sh)
    ap = nn.ParameterList([nn.Parameter(p, requires_grad=False) for p in ap_parts])

    # 4b Arm 实例
    arm = Arm(arm_dim, n_hidden)
    arm.initialize_parameters_map()
    arm.eval()
    laplace_model = constriction.stream.model.QuantizedLaplace(-LAPLACE_RANGE, LAPLACE_RANGE)

    # 4c 解码 latent（逐 level，逐图，逐像素流式）
    print(f"[4] decoding latent (ARM streaming CABAC) ...")
    dec_grids = []
    n_total = DECODE_N_IMAGES if DECODE_N_IMAGES is not None else batch
    print(f"    {n_grids} levels × {n_total} images = {n_grids * n_total} total")
    t0 = time.time()
    for lvl, sz in enumerate(h["latent_shapes"]):
        batch_l, _, H_l, W_l = sz
        raw = read_bin(os.path.join(BITSTREAM_DIR, f"latent_l{lvl}.bin"))
        ctx_lookup = make_context_lookup(H_l, W_l, mask_size, arm_dim)
        n_imgs = DECODE_N_IMAGES if DECODE_N_IMAGES is not None else batch_l
        sent_l = decode_latent_level_fast(raw, H_l, W_l, ctx_lookup, arm, ap, laplace_model, n_imgs)
        # 如果 batch_l > n_imgs, 后面补零
        if n_imgs < batch_l:
            pad = torch.zeros(batch_l - n_imgs, 1, H_l, W_l)
            sent_l = torch.cat([sent_l, pad], dim=0)
        dec_grids.append(sent_l / eg)
    print(f"    done in {time.time()-t0:.1f}s")

    # [5] 重建 DPParams
    print(f"[5] rebuilding DPParams ...")
    rebuilt_pool = {}
    for mod_name, pool_key in {"arm": "ap", "upsampling": "up", "synthesis": "sp"}.items():
        m = nn_dec[mod_name]["meta"]
        qsw, qsb = m["q_step_w"], m["q_step_b"]
        sw_all, sb_all = nn_dec[mod_name]["weight"], nn_dec[mod_name]["bias"]
        parts = []; iw = ib = 0
        for lm in m["layers"]:
            sh, n, is_w = lm["shape"], lm["numel"], lm["is_weight"]
            if is_w:
                parts.append(sw_all[iw:iw+n].float() * qsw); iw += n
            else:
                parts.append(sb_all[ib:ib+n].float() * qsb); ib += n
            parts[-1] = parts[-1].view(*sh)
        rebuilt_pool[pool_key] = parts
    rebuilt_pool["grids"] = dec_grids

    dp = DPParams()
    for pk in ("grids", "ap", "up", "sp"):
        dp.pool[pk] = nn.ParameterList([nn.Parameter(p.to(DEVICE), requires_grad=False) for p in rebuilt_pool[pk]])
    out_pt = os.path.join(OUT_DIR, "decoded.pt")
    torch.save({"param": dp}, out_pt)
    print(f"    -> {out_pt}  ({os.path.getsize(out_pt):,} bytes)")

    # [6] forward 出图
    print(f"[6] forward and dumping PNGs ...")
    op = TensorData(image_size=(H, W), channel=ch, device=DEVICE,
                    version=h["layers_v"], arm=arm_dim, dim=n_hidden)
    op.set_param(dp); op.set_to_eval()
    with torch.no_grad():
        y, _ = op.gen.forward(dp.pool["grids"], dp.pool["ap"], dp.pool["up"], dp.pool["sp"],
                              quantizer_noise_type="none", quantizer_type="hardround")
    max_dyn = 2 ** (op.param.bitdepth) - 1
    imgs = (torch.round(y * max_dyn) / max_dyn).clamp(0, 1).detach().cpu()
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(img_dir, f"img_{i:02d}.png"))
    grid = vutils.make_grid(imgs, nrow=int(math.ceil(math.sqrt(imgs.shape[0]))), normalize=False)
    arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(os.path.join(img_dir, "grid.png"))
    print(f"    -> {img_dir}/")
    print(f"\n[7] DONE  (bitstream {total/1024:.2f} KB → {imgs.shape[0]} images)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="entropy_codec decode_v2: bitstream -> decoded.pt + images")
    p.add_argument("--bitstream_dir", type=str, default=BITSTREAM_DIR)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--laplace_range", type=int, default=LAPLACE_RANGE)
    p.add_argument("--n_images", type=int, default=DECODE_N_IMAGES,
                   help="只解前 N 张图（调试）；不传则全量解码")
    args = p.parse_args()
    BITSTREAM_DIR = args.bitstream_dir
    OUT_DIR = args.out_dir
    LAPLACE_RANGE = args.laplace_range
    DECODE_N_IMAGES = args.n_images
    main()
