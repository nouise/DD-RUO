# entropy_codec（可选工具）

把 TensorPool 的 `pool.pt` 熵编码为紧凑二进制码流，并支持解码与分析。与主 pipeline 解耦，仅此工具需要额外依赖。

## 依赖

需要 `torchac` 与 `constriction` 两个熵编码库（主 pipeline 不需要）。

## 用法

在脚本顶部修改 `POOL_PATH` 等配置。其中 `LAYERS_V` / `ARM_DIM` / `N_HIDDEN_ARM` / `MSE_ERR` / `MASK_SIZE` 必须与 pool 训练时的设置一致。

```bash
cd entropy_codec

# 1. 编码：pool.pt → bitstream_out/
bash scripts/run_encode.sh

# 2. 解码：bitstream_out/ → decoded_out/
bash scripts/run_decode.sh

# 3. 分析：bpp / PSNR / 对比图 → analysis_out/
bash scripts/run_analyze.sh
```

也可直接调用 Python（参数走 argparse）：

```bash
python encode_v2.py --pool_path /path/to/pool.pt --layers_v v5 --arm_dim 32 --mask_size 9
python decode_v2.py --bitstream_dir ./bitstream_out
python analyze.py   --pool_path /path/to/pool.pt --bitstream_dir ./bitstream_out
```

## 输出

| 目录 | 内容 |
|---|---|
| `bitstream_out/` | 编码产出的二进制码流（`.bin`）|
| `decoded_out/` | 解码出的 `decoded.pt` 与合成图 PNG |
| `analysis_out/` | 分析报告（对比图 + `summary.txt`）|
