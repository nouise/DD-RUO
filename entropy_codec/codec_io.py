"""
header.bin 二进制布局（little-endian）

说明：
  - 不预存每 grid level 的 (lo, hi, mu, sigma)，ARM 解码时现场计算
  - mask_size: uint8（ARM context mask 边长，需与 pool 训练时一致）
  - Magic = "TMC2"

Layout
------
Magic         "TMC2"                          4B
version       uint8                           1B   (=1)
channel       uint16                          2B
batch         uint16                          2B
H, W          uint16, uint16                  4B
arm_dim       uint16                          2B
n_hidden_arm  uint8                           1B
mask_size     uint8                           1B    （ARM context mask 边长）
layers_v      pascal-string (uint8 len + bytes)
encoder_gain  float32                         4B
n_grids       uint8                           1B
  for each grid: B,C,H,W  uint16×4            8B
  (不存每 level 的统计量，ARM 解码时现场算)
n_modules     uint8                           1B  (=3)
  for each module:
    name      pascal-string
    n_layers  uint8
      for each layer:
        ndim    uint8
        dims    uint16 × ndim
        is_w    uint8  (1=weight, 0=bias)
    q_step_w  float32
    q_step_b  float32
    for kind in ("weight","bias"):
      n         int32     (-1 表示空)
      if n>0:
        lo      int32
        hi      int32
        sigma   float32

也提供 latent_l*.bin / nn_*.bin 用的简单容器：
    [uint32 n_bytes][bytes]
"""
import struct
from io import BytesIO


MAGIC = b"TMC2"
VERSION = 1


# ------------ 基础读写 ------------
def _w_u8(buf, v):  buf.write(struct.pack("<B", v))
def _w_u16(buf, v): buf.write(struct.pack("<H", v))
def _w_u32(buf, v): buf.write(struct.pack("<I", v))
def _w_i32(buf, v): buf.write(struct.pack("<i", v))
def _w_f32(buf, v): buf.write(struct.pack("<f", float(v)))
def _w_str(buf, s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    if len(s) > 255:
        raise ValueError("string too long for pascal encoding")
    buf.write(struct.pack("<B", len(s)))
    buf.write(s)


def _r_u8(buf):  return struct.unpack("<B", buf.read(1))[0]
def _r_u16(buf): return struct.unpack("<H", buf.read(2))[0]
def _r_u32(buf): return struct.unpack("<I", buf.read(4))[0]
def _r_i32(buf): return struct.unpack("<i", buf.read(4))[0]
def _r_f32(buf): return struct.unpack("<f", buf.read(4))[0]
def _r_str(buf):
    n = _r_u8(buf)
    return buf.read(n).decode("utf-8")


# ------------ header ------------
def write_header(path, header):
    """header dict 见下方 keys。"""
    buf = BytesIO()
    buf.write(MAGIC)
    _w_u8(buf, VERSION)
    _w_u16(buf, header["channel"])
    _w_u16(buf, header["batch"])
    _w_u16(buf, header["im_size"][0])
    _w_u16(buf, header["im_size"][1])
    _w_u16(buf, header["arm_dim"])
    _w_u8(buf, header["n_hidden_arm"])
    _w_u8(buf, header["mask_size"])
    _w_str(buf, header["layers_v"])
    _w_f32(buf, header["encoder_gain"])

    grids = header["latent_shapes"]
    _w_u8(buf, len(grids))
    for sh in grids:
        # 期望 4 维 (B,C,H,W)
        for d in sh:
            _w_u16(buf, int(d))
    # 不存 latent_levels 统计量

    modules = header["modules"]  # list of dicts
    _w_u8(buf, len(modules))
    for m in modules:
        _w_str(buf, m["name"])
        layers = m["layers"]
        _w_u8(buf, len(layers))
        for lm in layers:
            sh = lm["shape"]
            _w_u8(buf, len(sh))
            for d in sh:
                _w_u16(buf, int(d))
            _w_u8(buf, 1 if lm["is_weight"] else 0)
        _w_f32(buf, m["q_step_w"])
        _w_f32(buf, m["q_step_b"])
        for kind in ("weight", "bias"):
            mk = m[kind]
            n = mk["n"]
            if n == 0:
                _w_i32(buf, -1)
            else:
                _w_i32(buf, n)
                _w_i32(buf, mk["lo"])
                _w_i32(buf, mk["hi"])
                _w_f32(buf, mk["sigma"])

    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def read_header(path):
    with open(path, "rb") as f:
        buf = BytesIO(f.read())
    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    version = _r_u8(buf)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    h = {}
    h["channel"]      = _r_u16(buf)
    h["batch"]        = _r_u16(buf)
    H = _r_u16(buf); W = _r_u16(buf)
    h["im_size"]      = (H, W)
    h["arm_dim"]      = _r_u16(buf)
    h["n_hidden_arm"] = _r_u8(buf)
    h["mask_size"]    = _r_u8(buf)
    h["layers_v"]     = _r_str(buf)
    h["encoder_gain"] = _r_f32(buf)

    n_grids = _r_u8(buf)
    grids = []
    for _ in range(n_grids):
        grids.append((_r_u16(buf), _r_u16(buf), _r_u16(buf), _r_u16(buf)))
    h["latent_shapes"] = grids
    # 不读 latent_levels

    n_modules = _r_u8(buf)
    modules = []
    for _ in range(n_modules):
        m = {"name": _r_str(buf)}
        n_layers = _r_u8(buf)
        layers = []
        for _ in range(n_layers):
            ndim = _r_u8(buf)
            sh = tuple(_r_u16(buf) for _ in range(ndim))
            is_w = bool(_r_u8(buf))
            layers.append({"shape": sh, "is_weight": is_w,
                           "numel": int(__import__('numpy').prod(sh))})
        m["layers"] = layers
        m["q_step_w"] = _r_f32(buf)
        m["q_step_b"] = _r_f32(buf)
        for kind in ("weight", "bias"):
            n = _r_i32(buf)
            if n < 0:
                m[kind] = {"n": 0}
            else:
                lo = _r_i32(buf)
                hi = _r_i32(buf)
                sigma = _r_f32(buf)
                m[kind] = {"n": n, "lo": lo, "hi": hi, "sigma": sigma}
        modules.append(m)
    h["modules"] = modules
    return h


# ------------ bin 容器 ------------
def write_bin(path, byte_stream):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(byte_stream)))
        f.write(byte_stream)


def read_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        return f.read(n)
