"""Issue #37: the same seeded sampled request through the loop that reads every pass (eng.mtp_pipelined=False) and the
pipelined one, on the real model, in process; and greedy MTP off against the pipelined loop. Prints the first difference.
Run from the repo root: .venv/bin/python docs/measurements/mtp/seeded_check.py <model> <mtp file>
"""
import sys, time, mlx.core as mx
sys.path.insert(0, "tests")
from test_mtp_model import PROMPTS
from locals.runtime import Runtime
from locals.hybrid_batch import HybridBatchEngine
from locals.engine import Controller
from locals.packed import Request
from locals.mtp import MTP
model, mtp_path = sys.argv[1], sys.argv[2]
rt = Runtime.load(model)
tok = rt.tokenizer
class Free(Controller):
    eos_ids = list(rt.vocab.eos_ids)
mtp = MTP.load(rt.model, mtp_path, drafts=1)
def run(p, use, pipelined, **kw):
    eng = HybridBatchEngine(rt.model, tok, n_slots=4, share_prefix=False, mtp=mtp if use else None)
    eng.mtp_pipelined = pipelined
    r = Request(list(p), Free(), 192, **kw); w = [r]
    t = time.perf_counter(); eng.serve(lambda b: [w.pop()] if w else []); dt = time.perf_counter() - t
    return r.out, eng.counters, dt
def diff(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return i if i is not None or len(a) == len(b) else min(len(a), len(b))
for name, text, tools in PROMPTS[:10]:
    p = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
    off, _, t_off = run(p, False, True)
    new, c, t_new = run(p, True, True)
    old, c_old, t_old = run(p, True, False)
    print(f"{name:14s} greedy: off/new first diff {diff(off, new)} ({len(off)} tok) old/new {diff(old, new)} | mtp {c.get('draft_mtp_accepted',0)}/{c.get('draft_mtp_tokens',0)} passes {c.get('stretch_passes',0)} discarded {c.get('stretch_discarded',0)} | tok/s off {len(off)/t_off:.0f} old {len(old)/t_old:.0f} new {len(new)/t_new:.0f}", flush=True)
    for seed in (1, 2):
        kw = {"temp": 0.7, "seed": seed}
        old, _, _ = run(p, True, False, **kw)
        new, c, _ = run(p, True, True, **kw)
        print(f"{'':14s} seed {seed}: old/new first diff {diff(old, new)} ({len(old)} vs {len(new)} tok) mtp {c.get('draft_mtp_accepted',0)}/{c.get('draft_mtp_tokens',0)}", flush=True)
