"""Usage: stage_cost.py MODEL — per request with two images: vision.load (HTTP thread), Encoder.encode (the tower),
Engine.prefill_embedded, and the whole request."""
import contextlib
import statistics
import sys
import time

import mlx.core as mx

import locals as L
from locals import engine as eng
from locals import vision
from locals.server import Server

import request

model_id, mode = sys.argv[1], "mrope"
print("locals from", L.__file__, "mode", mode, flush=True)
if mode == "stock":
    vision.mrope = lambda *a, **k: contextlib.nullcontext()
swapped = []
real_ctx = vision.mrope
def counting(model, positions, section):
    cm = real_ctx(model, positions, section)
    class W:
        def __enter__(s):
            r = cm.__enter__()
            swapped.append(sum(isinstance(getattr(getattr(l, "self_attn", None), "rope", None), vision.MRoPE)
                               for l in eng.text_model(model).model.layers))
            return r
        def __exit__(s, *e): return cm.__exit__(*e)
    return W()
vision.mrope = counting
rec = []
real = eng.Engine.prefill_embedded
def timed(self, cache, prompt, *a, **k):
    mx.synchronize(); t = time.perf_counter()
    r = real(self, cache, prompt, *a, **k)
    mx.synchronize(); rec.append((len(prompt), time.perf_counter() - t)); return r
eng.Engine.prefill_embedded = timed
stages = {"load": [], "tower": []}
vision.load = request.timer(stages, "load", vision.load, False)
vision.Encoder.encode = request.timer(stages, "tower", vision.Encoder.encode, True)

req = request.request()
srv = Server(model_id, engines=("locals",), preload=("locals",))
whole = []
for _ in range(4):
    t = time.perf_counter(); srv.complete(req); whole.append(time.perf_counter() - t)
n = rec[-1][0]; ts = [t for _, t in rec[1:]]
med = lambda v: statistics.median(v[1:]) * 1000
print(f"{model_id}: load {med(stages['load']):.0f} ms, tower {med(stages['tower']):.0f} ms, prefill {med([t for _, t in rec]):.0f} ms, "
      f"whole request {med(whole):.0f} ms (medians of runs 2-4)", flush=True)
print(f"{model_id} {mode}: prompt {n} tokens, prefill_embedded median {statistics.median(ts)*1000:.0f} ms "
      f"(runs 2-4: {', '.join(f'{t*1000:.0f}' for t in ts)}), layers on MRoPE: {swapped[-1]}", flush=True)
srv.unload()
