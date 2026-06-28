"""
analyze.py — 编/解码结果分析

输入：
    POOL_PATH               原 pool .pt
    BITSTREAM_DIR           encode.py 的输出
    DECODED_PT (可选)       decode.py 输出的 decoded.pt（不存在则自动跑一次解码）

输出（写入 ANALYSIS_DIR）：
    compare_3x3.png         3 行 × 3 列 对比图（量化后 fwd | 解码后 fwd | 差值×10）
    summary.txt             文字摘要：bpp / PSNR / 各张量 max_abs_diff / 文件清单

要求：encode.py 已跑过；解码端会被本脚本拉起一次以拿到 fwd 后的图。
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

import core  # noqa: F401  (triggers the ts->core.ts shim in core/__init__.py)

from core.ts.tensor_data_func_v6 import TensorData
from core.ts.core.quantizemodel import quantize_model_no_ref_v2
from core.ts.core.arm_func import _get_neighbor

# 复用 encode_v2.py 里的工具
from encode_v2 import (
    POOL_PATH, OUT_DIR as BITSTREAM_DIR, DEVICE, LAYERS_V,
    ARM_DIM, N_HIDDEN_ARM, MSE_ERR, SLICE_INDEX
)


ANALYSIS_DIR = os.path.join(THIS_DIR, "analysis_out")
SHOW_INDICES = [0, 25, 50]


@torch.no_grad()
def forward_image(op):
    op.set_to_eval()
    y, _ = op.gen.forward(
        op.dp.pool["grids"], op.dp.pool["ap"], op.dp.pool["up"], op.dp.pool["sp"],
        quantizer_noise_type="none", quantizer_type="hardround",
    )
    max_dyn = 2 ** (op.param.bitdepth) - 1
    return (torch.round(y * max_dyn) / max_dyn).clamp(0.0, 1.0).detach().cpu()


def psnr(a, b, max_val=1.0):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse <= 0 else 10.0 * math.log10((max_val ** 2) / mse)


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # [1] 拿到「编码前」（量化后 fwd 的图）+ 「编码前」的 dp（量化后 dp）
    print(f"[1] reload pool and re-quantize to get 'pre-encode' state ...")
    slice_pool = torch.load(POOL_PATH, map_location="cpu", weights_only=False)
    keys = list(slice_pool.keys())
    pk = keys[SLICE_INDEX]
    dp_pre = slice_pool[pk]["param"]
    dp_pre.load_reset()
    grids = dp_pre.pool["grids"]
    batch, _, H, W = grids[0].shape
    im_size = (H, W)
    channel = dp_pre.pool["sp"][4].shape[0]

    op = TensorData(image_size=im_size, channel=channel, device=DEVICE,
                    version=LAYERS_V, arm=ARM_DIM, dim=N_HIDDEN_ARM)
    op.set_param(dp_pre)
    op.to_run()
    best_bpp, distribution, quant_param = quantize_model_no_ref_v2(op, dp_pre, mse_err=MSE_ERR)
    op.set_param(dp_pre)
    img_pre = forward_image(op)

    # [2] 拉起 decoder（保证 decoded.pt + 图都存在）
    decoded_pt = os.path.join(THIS_DIR, "decoded_out", "decoded.pt")
    if not os.path.exists(decoded_pt):
        print(f"\n[2] decoded.pt not found, running decode.py ...")
        import decode as _decode
        _decode.main()
    else:
        print(f"\n[2] reuse existing {decoded_pt}")
    dec_blob = torch.load(decoded_pt, map_location="cpu", weights_only=False)
    dp_post = dec_blob["param"]
    # 把 dp_post 的所有参数搬到 op 当前所在 device
    for pk_ in ("grids", "ap", "up", "sp"):
        new_pl = nn.ParameterList([
            nn.Parameter(p.detach().to(DEVICE), requires_grad=False)
            for p in dp_post.pool[pk_]
        ])
        dp_post.pool[pk_] = new_pl

    # [3] 解码后 forward
    print(f"\n[3] decoded-side forward ...")
    op.set_param(dp_post)
    img_post = forward_image(op)
    p = psnr(img_pre, img_post, max_val=1.0)
    print(f"    PSNR(pre vs post) = {p:.2f} dB")

    # [3b] ARM 估计 bpp（理论极限） vs 实际 bpp
    print(f"\n[3b] ARM-estimate bpp (theory) vs actual bpp ...")
    from core.ts.core.arm_func import _laplace_cdf as laplace_cdf_fn
    op.set_param(dp_pre)  # 用量化后参数（与 encode 端一致）
    op.set_to_eval()
    gen = op.gen
    encoder_gain_t = gen.encoder_gains
    with torch.no_grad():
        size_per_latent_flat = [l.numel() for l in dp_pre.pool["grids"]]
        size_per_latent = [l.shape for l in dp_pre.pool["grids"]]
        encoder_side_flat = torch.cat([l.view(-1) for l in dp_pre.pool["grids"]]).contiguous()
        flat_dec = torch.round(encoder_side_flat * encoder_gain_t)
        dsl = [t.view(sz) for t, sz in zip(torch.split(flat_dec, size_per_latent_flat), size_per_latent)]
        flat_ctx = torch.cat([_get_neighbor(sl, gen.mask_size, gen.non_zero_pixel_ctx_index)
                              for sl in dsl], dim=0)
        flat_mu, flat_sc, _ = gen.arm(flat_ctx, dp_pre.pool["ap"])
        proba = torch.clamp_min(
            laplace_cdf_fn(flat_dec + 0.5, flat_mu, flat_sc)
            - laplace_cdf_fn(flat_dec - 0.5, flat_mu, flat_sc),
            min=2**-16,
        )
        flat_rate = -torch.log2(proba)
    arm_bpp = flat_rate.sum().item() / (dp_pre.pool["grids"][0].shape[0]
                                         * dp_pre.pool["grids"][0].shape[-2]
                                         * dp_pre.pool["grids"][0].shape[-1])
    print(f"    ARM-estimate latent bpp = {arm_bpp:.4f}  (theoretical lower bound)")

    # [4] 参数 diff（grids 比 sent 整数；ap/up/sp 比浮点）
    print(f"\n[4] decoded params vs original (post-quantize) params ...")
    encoder_gain = op.gen.encoder_gains
    eg_val = float(encoder_gain) if not torch.is_tensor(encoder_gain) or encoder_gain.numel() == 1 \
             else float(encoder_gain.flatten()[0])
    rows = []
    rows.append(f"{'module':<14s}{'tensor':<10s}{'numel':>10s}{'max_abs_diff':>20s}")
    total_max_diff = 0.0

    # grids（比 sent 整数）：pre 的 grids 是 fp，乘以 gain 取整就是 sent_pre；
    # post 的 grids = sent_post / gain → 乘以 gain 取整就是 sent_post。
    for i, (g_pre, g_post) in enumerate(zip(dp_pre.pool["grids"], dp_post.pool["grids"])):
        s_pre = torch.round(g_pre.detach().cpu().to(torch.float32) * eg_val).to(torch.long)
        s_post = torch.round(g_post.detach().cpu().to(torch.float32) * eg_val).to(torch.long)
        diff = (s_pre - s_post).abs().max().item()
        total_max_diff = max(total_max_diff, float(diff))
        rows.append(f"{'grids(sent)':<14s}[{i}]   {s_pre.numel():>10d}{diff:>20.3e}")

    # 网络（比浮点参数 — dp_pre.pool[ap/up/sp] 已被 quantize 替换为 q_param）
    for pk_ in ("ap", "up", "sp"):
        for i, (orig, dec) in enumerate(zip(dp_pre.pool[pk_], dp_post.pool[pk_])):
            o = orig.detach().cpu().to(torch.float32)
            d = dec.detach().cpu().to(torch.float32)
            diff = (o - d).abs().max().item()
            total_max_diff = max(total_max_diff, diff)
            rows.append(f"{pk_:<14s}[{i}]   {o.numel():>10d}{diff:>20.3e}")
    verdict = "LOSSLESS bitstream" if total_max_diff < 1e-6 else "LOSSY bitstream (bug?)"
    rows.append(f">>> overall max_abs_diff = {total_max_diff:.3e}  ({verdict})")
    for r in rows:
        print("    " + r)

    # [5] 3×3 对比图
    print(f"\n[5] saving 3x3 comparison figure ...")
    pre = img_pre.clamp(0, 1)
    post = img_post.clamp(0, 1)
    diff = ((pre - post).abs() * 10.0).clamp(0, 1)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    col_titles = ["before encoding\n(quantized fwd)",
                  "after decoding\n(decoded fwd)",
                  "abs diff (×10)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)
    for row, idx in enumerate(SHOW_INDICES):
        for col, src in enumerate([pre, post, diff]):
            ax = axes[row, col]
            ax.imshow(src[idx].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"img #{idx}", fontsize=9)
    fig.suptitle(f"slice={pk}  PSNR(pre vs post)={p:.2f} dB", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, "compare_3x3.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"    -> {fig_path}")

    # [6] summary.txt
    print(f"\n[6] writing summary ...")
    file_records = []
    for fn in sorted(os.listdir(BITSTREAM_DIR)):
        if not fn.endswith(".bin"):
            continue
        file_records.append((fn, os.path.getsize(os.path.join(BITSTREAM_DIR, fn))))
    bs_total = sum(b for _, b in file_records)
    pool_size = os.path.getsize(POOL_PATH)
    raw_per_slice = pool_size / len(keys)
    npix = batch * im_size[0] * im_size[1]

    actual_bpp = bs_total * 8 / npix
    ratio_actual_vs_arm = actual_bpp / max(arm_bpp, 1e-9)
    summary_lines = [
        f"# slice={pk}",
        f"pool.pt           : {pool_size:>12,d} bytes ({pool_size/1024/1024:.2f} MB)",
        f"raw / slice       : {raw_per_slice:>12,.0f} bytes ({raw_per_slice/1024/1024:.2f} MB)",
        f"bitstream total   : {bs_total:>12,d} bytes ({bs_total/1024:.2f} KB)",
        f"compression ratio : {raw_per_slice/max(bs_total,1):>12.2f}x  "
        f"(saved {(1-bs_total/raw_per_slice)*100:.2f}%)",
        f"bpp (actual total): {actual_bpp:.6f}",
        f"bpp (ARM-est, latent only, theory): {arm_bpp:.6f}",
        f"bpp (ARM-est, total, from quantize_net): {best_bpp:.6f}",
        f"actual / ARM-est-total ratio: {actual_bpp/max(best_bpp,1e-9):.4f}",
        f"PSNR (pre vs post): {p:.2f} dB",
        f"verdict           : {verdict}",
        f"",
        f"# per-file breakdown",
    ]
    for fn, nb in file_records:
        summary_lines.append(f"  {fn:<35s}{nb:>10,d} bytes ({nb/max(bs_total,1)*100:5.2f}%)")
    summary_lines.append("")
    summary_lines.append("# parameter diff (decoded vs post-quantize)")
    summary_lines.extend(rows)

    summary_path = os.path.join(ANALYSIS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    for ln in summary_lines:
        print("    " + ln)
    print(f"\n[7] DONE -> {ANALYSIS_DIR}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="entropy_codec analyze: compare original pool vs bitstream vs decoded")
    p.add_argument("--pool_path", type=str, default=POOL_PATH)
    p.add_argument("--bitstream_dir", type=str, default=BITSTREAM_DIR)
    p.add_argument("--analysis_dir", type=str, default=ANALYSIS_DIR)
    args = p.parse_args()
    POOL_PATH = args.pool_path
    BITSTREAM_DIR = args.bitstream_dir
    ANALYSIS_DIR = args.analysis_dir
    main()
