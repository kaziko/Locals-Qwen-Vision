"""Usage: patch_cost.py MODEL [OUT.npz] — the parts issue #26 names, timed inside the request stage_cost.py sends
(text, a 1280x800 and a 1024x1024 picture, max_tokens 1): vision.preprocess (HTTP thread, numpy), the tower's
patch embed alone (its conv, evaluated on its own), vision.positions (the prompt's M-RoPE positions, evaluated), the
whole tower, and the whole request. Medians of runs 2-4. With OUT.npz, the last run's tower features and positions
are saved, so two builds of the code can be compared value for value."""
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

import locals as L
from locals import vision
from locals.models import qwen3_5_vl as q
from locals.server import Server

import request

model_id = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
print("locals from", L.__file__, flush=True)

stages = {"preprocess": [], "patch_embed": [], "positions": [], "tower": []}
saved = {"features": [], "positions": None}

vision.preprocess = request.timer(stages, "preprocess", vision.preprocess, False)
q.PatchEmbed.__call__ = request.timer(stages, "patch_embed", q.PatchEmbed.__call__, True)
timed_positions = request.timer(stages, "positions", vision.positions, True)
timed_encode = request.timer(stages, "tower", vision.Encoder.encode, True)  # encode evals its outputs itself: the eval here is a no-op


def encode(self, pictures):
    feats, deep = timed_encode(self, pictures)
    saved["features"] = [np.array(f.astype(mx.float32)) for f in feats]
    return feats, deep


vision.Encoder.encode = encode


def positions(*a, **k):
    r = timed_positions(*a, **k)
    saved["positions"] = np.array(r)
    return r


vision.positions = positions

req = request.request()
srv = Server(model_id, engines=("locals",), preload=("locals",))
whole = []
for _ in range(4):
    request.empty_cache(srv)  # every run runs the tower
    t = time.perf_counter()
    srv.complete(req)
    whole.append(time.perf_counter() - t)

for key, v in stages.items():  # a hook that never fired, or fired unevenly, is an error, not a 0.00 ms stage
    assert v and len(v) % 4 == 0, f"{key}: {len(v)} call(s) over 4 requests"
assert saved["positions"] is not None and saved["features"], "no positions or features captured"


def med(v):  # the sum over a request (the calls split evenly over the 4 requests), median of runs 2-4
    per = len(v) // 4
    sums = [sum(v[i * per:(i + 1) * per]) for i in range(4)]
    return statistics.median(sums[1:]) * 1000, [s * 1000 for s in sums], per


lines = []
for key, v in stages.items():
    m, runs, per = med(v)
    lines.append(f"{key} {m:.2f} ms (runs: {', '.join(f'{r:.2f}' for r in runs)}; {per} call(s) a request)")
m_whole = statistics.median(whole[1:]) * 1000
print(f"{model_id}: " + ", ".join(lines) + f", whole request {m_whole:.0f} ms (runs: {', '.join(f'{r * 1000:.0f}' for r in whole)})", flush=True)
print(f"positions shape {saved['positions'].shape}, features " + ", ".join(str(f.shape) for f in saved["features"]), flush=True)
if out:
    np.savez(out, positions=saved["positions"], **{f"features{i}": f for i, f in enumerate(saved["features"])})
    print("saved", out, flush=True)
srv.unload()
