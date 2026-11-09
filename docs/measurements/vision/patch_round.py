"""Usage: patch_round.py MODEL — how far the patch embed's output moved (issue #26), at the patch embed itself, on
the two pictures stage_cost.py sends. Three outputs over the same pixels: the checkpoint's conv over two frames in
bfloat16 (the code before), the summed-weight matmul in bfloat16 (the code now, the loaded tower's), and the conv in
float32 (the exact value both round). Each bfloat16 output's max abs difference from the float32 one, the two
bfloat16 outputs' from each other, and bfloat16's spacing at the largest output, for scale."""
import base64
import io
import sys

import mlx.core as mx
import numpy as np
from PIL import Image

import locals as L
from locals import vision
from locals.models import qwen3_5_vl as q

model_id = sys.argv[1]
print("locals from", L.__file__, flush=True)
enc = vision.Encoder(model_id, mx.bfloat16)
tower = enc.tower()
d = enc.directory()
cfg = enc._config
w5 = [v for k, v in q._weights(d) if k == "patch_embed.proj.weight"][0]
if w5.shape[1] == cfg.in_channels and w5.shape[-1] != cfg.in_channels:
    w5 = w5.transpose(0, 2, 3, 4, 1)  # transformers' layout -> MLX's (out, frames, h, w, channels)
b = [v for k, v in q._weights(d) if k == "patch_embed.proj.bias"][0]
print(f"conv weight {w5.shape} {w5.dtype}, bias {b.shape} {b.dtype}", flush=True)

rng = np.random.default_rng(0)


def pic(w, h):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1).astype(np.int16)
    a = np.clip(a + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def conv(x1, dtype):
    n = x1.shape[0]
    x2 = mx.broadcast_to(x1.reshape(n, 1, 16, 16, 3), (n, 2, 16, 16, 3)).astype(dtype)
    return mx.conv3d(x2, w5.astype(dtype), stride=(2, 16, 16)).reshape(n, -1) + b.astype(dtype)


for raw in (pic(1280, 800), pic(1024, 1024)):
    p = vision.preprocess(raw)
    x1 = mx.array(p.patches)  # float32 pixels
    exact = conv(x1, mx.float32)
    before = conv(x1, mx.bfloat16).astype(mx.float32)
    now = tower.patch_embed(x1.astype(mx.bfloat16)).astype(mx.float32)
    mx.eval(exact, before, now)
    e, bf, nw = (np.array(t).astype(np.float64) for t in (exact, before, now))
    top = np.abs(e).max()
    ulp = 2.0 ** (np.floor(np.log2(top)) - 7)  # bfloat16: 8 bits of mantissa (7 stored)
    print(f"grid {p.grid}, {p.patches.shape[0]} patches, hidden {e.shape[1]}: largest exact output {top:.4g} "
          f"(bfloat16 spacing there {ulp:.4g}); max abs diff from exact: two-frame conv in bfloat16 {np.abs(bf - e).max():.4g}, "
          f"summed matmul in bfloat16 {np.abs(nw - e).max():.4g}; between the two {np.abs(bf - nw).max():.4g} "
          f"(mean {np.abs(bf - nw).mean():.4g})", flush=True)
