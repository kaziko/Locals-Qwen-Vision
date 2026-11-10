"""Issue #37: mx.async_eval per pass of the pipelined loop with parts of the pass replaced by cheaper stand-ins (timing
only: the replies are wrong then): the recurrent layers' state selection, their per-token states, the MTP layer's fill.
Run from the repo root: HF_HOME=... LOCALS_DISK_CACHE=0 .venv/bin/python docs/measurements/mtp/stream_ablation.py <model> <mtp file>
"""
import sys, time, mlx.core as mx
sys.path.insert(0, "tests")
from test_mtp_model import PROMPTS
from locals.runtime import Runtime
from locals import hybrid_batch as hb
from locals.hybrid_batch import HybridBatchEngine
from locals.engine import Controller
from locals.packed import Request
from locals.mtp import MTP
import statistics as st
model, mtp_path = sys.argv[1], sys.argv[2]
rt = Runtime.load(model); tok = rt.tokenizer
class Free(Controller):
    eos_ids = list(rt.vocab.eos_ids)
mtp = MTP.load(rt.model, mtp_path, drafts=1)
T = []
orig_async = mx.async_eval
def timed_async(*a, **k):
    t = time.perf_counter(); out = orig_async(*a, **k); T.append(time.perf_counter() - t); return out
hb.mx.async_eval = timed_async
p = tok.encode(tok.apply_chat_template([{"role": "user", "content": PROMPTS[1][1]}], add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
def run(use, n=96):
    eng = HybridBatchEngine(rt.model, tok, n_slots=4, share_prefix=False, mtp=mtp if use else None)
    r = Request(list(p), Free(), n); w = [r]
    T.clear(); t = time.perf_counter(); eng.serve(lambda b: [w.pop()] if w else []); dt = time.perf_counter() - t
    return f"{len(r.out)/dt:.0f} tok/s async_eval median {1000*st.median(T):.2f} ms x{len(T)}"
run(False)
print("plain          ", run(False))
print("stream         ", run(True))
# ablation 1: no device GDN select (states kept as the kernel left them: after both tokens)
o = hb._gdn_at
hb._gdn_at = lambda c, ts, conv, keep: None
print("no gdn select  ", run(True))
hb._gdn_at = o
# ablation 2: no per-token states recorded (want_states off): the kernel's plain path
o_launch = HybridBatchEngine._stream_launch
class NoStates:
    pass
def launch(self, ctx, cache, st_, slot):
    r, fa, rec, entry, *_ = ctx
    for c in rec: c.want_states = False
    o_model = self.lm.model.__call__
    out = o_launch(self, ctx, cache, st_, slot)
    return out
import types
orig_mix = type(rt.model.language_model.model.layers[0].linear_attn if hasattr(rt.model, "language_model") else rt.model.model.layers[0].linear_attn)._mix
def mix_nostates(self, proj, mask=None, cache=None):
    if cache is not None:
        cache.want_states = False
    out = orig_mix(self, proj, mask, cache)
    if cache is not None and getattr(cache, "token_states", None) is None and proj.shape[1] == 2:
        cache.token_states = mx.stack([cache.cache[1], cache.cache[1]])
        cache.token_conv = (cache.cache[0], proj[..., :self.conv_dim])
    return out
cls = type(rt.model.language_model.model.layers[0].linear_attn if hasattr(rt.model, "language_model") else rt.model.model.layers[0].linear_attn)
cls._mix = mix_nostates
print("no token states", run(True))
cls._mix = orig_mix
# ablation 3: no MTP fill (draft = a constant)
o_fill = MTP.fill
MTP.fill = lambda self, entry, hidden, nexts, flags, offset=None: (self.holes(entry, 1, hidden.shape[1]), hidden)[1]
print("no mtp fill    ", run(True))
MTP.fill = o_fill
