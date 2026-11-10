"""Issue #37: where a pass of the pipelined loop spends its time, in process, greedy: the CPU building the pass (Python),
mx.async_eval (encoding, and the wait for the pass before it to leave the GPU), the wait for its readback; and the
same for a plain step (MTP off). stream_ablation.py times the pass with parts of it skipped.
Run from the repo root: HF_HOME=... LOCALS_DISK_CACHE=0 .venv/bin/python docs/measurements/mtp/stream_profile.py <model> <mtp file>
"""
import sys, time, mlx.core as mx
sys.path.insert(0, "tests")
from test_mtp_model import PROMPTS
from locals.runtime import Runtime
from locals import hybrid_batch as hb, mtp as M
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
T = {}
def timed(name, f):
    def w(*a, **k):
        t = time.perf_counter(); out = f(*a, **k); T.setdefault(name, []).append(time.perf_counter() - t); return out
    return w
hb.mx.async_eval = timed("async_eval", mx.async_eval)
hb.head_logits = timed("head_logits", hb.head_logits)
hb._gdn_at = timed("gdn_at", hb._gdn_at)
MTP.fill = timed("fill", MTP.fill)
MTP.draft_on_device = timed("draft_dev", MTP.draft_on_device)
HybridBatchEngine._growth_trim = timed("growth_trim", HybridBatchEngine._growth_trim)
HybridBatchEngine._stream_launch = timed("launch", HybridBatchEngine._stream_launch)
o_commit = HybridBatchEngine._stream_commit
def commit(self, ctx, cur, dead, holes):
    t = time.perf_counter(); cur["rb"].tolist(); T.setdefault("readback wait", []).append(time.perf_counter() - t)
    return o_commit(self, ctx, cur, dead, holes)
HybridBatchEngine._stream_commit = commit
lm = hb.text_model(rt.model)
body = lm.model
orig_call = type(body).__call__
type(body).__call__ = timed("forward", orig_call)
p = tok.encode(tok.apply_chat_template([{"role": "user", "content": PROMPTS[1][1]}], add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
def run(use):
    eng = HybridBatchEngine(rt.model, tok, n_slots=4, share_prefix=False, mtp=mtp if use else None)
    r = Request(list(p), Free(), 192); w = [r]
    t = time.perf_counter(); eng.serve(lambda b: [w.pop()] if w else []); dt = time.perf_counter() - t
    return r, dt
run(False); T.clear(); r, dt = run(False)
print("plain", f"{len(r.out)/dt:.0f} tok/s", {k: f"{1000*st.median(v):.2f}ms x{len(v)}" for k, v in T.items()})
T.clear(); r, dt = run(True)
print("stream", f"{len(r.out)/dt:.0f} tok/s", {k: f"{1000*st.median(v):.2f}ms x{len(v)} (sum {1000*sum(v):.0f})" for k, v in T.items()})
