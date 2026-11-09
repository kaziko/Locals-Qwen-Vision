"""Usage: patch_cost.py MODEL [OUT.npz] — the parts issue #26 names, timed inside the request stage_cost.py sends
(text, a 1280x800 and a 1024x1024 picture, max_tokens 1): vision.preprocess (HTTP thread, numpy), the tower's
patch embed alone (its conv, evaluated on its own), vision.positions (the prompt's M-RoPE positions, evaluated), the
whole tower, and the whole request. Medians of runs 2-4. With OUT.npz, the last run's tower features and positions
are saved, so two builds of the code can be compared value for value."""
import base64
import io
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
from PIL import Image

import locals as L
from locals import vision
from locals.models import qwen3_5_vl as q
from locals.server import Server

model_id = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
print("locals from", L.__file__, flush=True)

stages = {"preprocess": [], "patch_embed": [], "positions": [], "tower": []}
saved = {"features": [], "positions": None}


def timer(key, f, sync):
    def w(*a, **k):
        if sync:
            mx.synchronize()
        t = time.perf_counter()
        r = f(*a, **k)
        if sync:
            mx.eval(r)
            mx.synchronize()
        stages[key].append(time.perf_counter() - t)
        return r
    return w


vision.preprocess = timer("preprocess", vision.preprocess, False)
q.PatchEmbed.__call__ = timer("patch_embed", q.PatchEmbed.__call__, True)
vision.positions = timer("positions", vision.positions, True)
real_encode = vision.Encoder.encode


def encode(self, pictures):
    mx.synchronize()
    t = time.perf_counter()
    feats, deep = real_encode(self, pictures)
    mx.synchronize()
    stages["tower"].append(time.perf_counter() - t)
    saved["features"] = [np.array(f.astype(mx.float32)) for f in feats]
    return feats, deep


vision.Encoder.encode = encode
real_positions = vision.positions


def positions(*a, **k):
    r = real_positions(*a, **k)
    saved["positions"] = np.array(r)
    return r


vision.positions = positions

rng = np.random.default_rng(0)


def pic(w, h):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1).astype(np.int16)
    a = np.clip(a + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}


filler = "".join(f"Line {i}: the parcel {i * 7919 % 10007} left the depot at {i % 24:02d}:{i * 13 % 60:02d}.\n" for i in range(150))
parts = [{"type": "text", "text": filler}, pic(1280, 800), pic(1024, 1024), {"type": "text", "text": "Describe both pictures."}]
srv = Server(model_id, engines=("locals",), preload=("locals",))
req = {"model": "locals", "max_tokens": 1, "temperature": 0, "messages": [{"role": "user", "content": parts}]}
whole = []
for _ in range(4):
    if srv.service("locals")._lru is not None:  # a cached prompt is read no further: empty the prompt cache first
        srv.service("locals")._lru.trim_to(n_sequences=0)
    t = time.perf_counter()
    srv.complete(req)
    whole.append(time.perf_counter() - t)

n_pics = len(stages["preprocess"]) // 4


def med(v, per):  # per: calls per request; the sum over a request, median of runs 2-4
    sums = [sum(v[i * per:(i + 1) * per]) for i in range(4)]
    return statistics.median(sums[1:]) * 1000, [s * 1000 for s in sums]


lines = []
for key, per in (("preprocess", n_pics), ("patch_embed", len(stages["patch_embed"]) // 4), ("positions", len(stages["positions"]) // 4), ("tower", len(stages["tower"]) // 4)):
    m, runs = med(stages[key], per)
    lines.append(f"{key} {m:.2f} ms (runs: {', '.join(f'{r:.2f}' for r in runs)}; {per} call(s) a request)")
m_whole = statistics.median(whole[1:]) * 1000
print(f"{model_id}: " + ", ".join(lines) + f", whole request {m_whole:.0f} ms (runs: {', '.join(f'{r * 1000:.0f}' for r in whole)})", flush=True)
if saved["positions"] is not None:
    print(f"positions shape {saved['positions'].shape}, features " + ", ".join(str(f.shape) for f in saved["features"]), flush=True)
if out:
    np.savez(out, positions=saved["positions"], **{f"features{i}": f for i, f in enumerate(saved["features"])})
    print("saved", out, flush=True)
srv.unload()
