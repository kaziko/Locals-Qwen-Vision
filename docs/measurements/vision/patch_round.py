"""Usage: patch_round.py MODEL — how far the patch embed's output moved (issue #26), at the patch embed itself, on
the two pictures stage_cost.py sends. Three outputs over the same pixels: the checkpoint's conv over two frames in
bfloat16 (the code before), the summed-weight matmul in bfloat16 (the code now, the loaded tower's), and the conv in
float32 (the exact value both round). Each bfloat16 output's max abs difference from the float32 one, the two
bfloat16 outputs' from each other, and bfloat16's spacing at the largest output, for scale."""
import sys

import mlx.core as mx
import numpy as np

import locals as L
from locals import vision
from locals.models import qwen3_5_vl as q

import request

model_id = sys.argv[1]
print("locals from", L.__file__, flush=True)
enc = vision.Encoder(model_id, mx.bfloat16)
tower = enc.tower()
d = enc.directory()
cfg = enc._config
ws = dict(q._weights(d))
w5 = q._conv_layout(ws["patch_embed.proj.weight"], cfg)  # MLX's (out, frames, h, w, channels)
b = ws["patch_embed.proj.bias"]
print(f"conv weight {w5.shape} {w5.dtype}, bias {b.shape} {b.dtype}", flush=True)

def conv(x1, dtype):
    n = x1.shape[0]
    x2 = mx.broadcast_to(x1.reshape(n, 1, 16, 16, 3), (n, 2, 16, 16, 3)).astype(dtype)
    return mx.conv3d(x2, w5.astype(dtype), stride=(2, 16, 16)).reshape(n, -1) + b.astype(dtype)


for raw in request.pictures():
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
