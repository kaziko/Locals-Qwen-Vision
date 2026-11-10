"""Issue #37: the repeated-file workload of tests/test_solo.py (prompt-lookup drafts' best case), greedy, in process:
MTP off (lookup drafts), MTP on with the loop that reads every pass, MTP on pipelined (no lookups inside a stretch).
Three rounds each, interleaved; prints tokens, tok/s and the draft counters.
Run from the repo root: .venv/bin/python docs/measurements/mtp/edit_workload.py <model> <mtp file>
"""
import sys, time, mlx.core as mx
sys.path.insert(0, "tests")
from test_solo import EDIT
from locals.runtime import Runtime
from locals.hybrid_batch import HybridBatchEngine
from locals.engine import Controller
from locals.packed import Request
from locals.mtp import MTP
model, mtp_path = sys.argv[1], sys.argv[2]
rt = Runtime.load(model)
class Free(Controller):
    eos_ids = list(rt.vocab.eos_ids)
mtp = MTP.load(rt.model, mtp_path, drafts=1)
p = rt.encode_chat(EDIT)
def run(use, pipelined):
    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4, share_prefix=False, mtp=mtp if use else None)
    eng.mtp_pipelined = pipelined
    r = Request(list(p), Free(), 300); w = [r]
    t = time.perf_counter(); eng.serve(lambda b: [w.pop()] if w else []); dt = time.perf_counter() - t
    c = eng.counters
    return r.out, len(r.out) / dt, {k: c.get(k, 0) for k in ("steps", "solo_steps", "drafted", "draft_lookup_tokens", "draft_lookup_accepted", "draft_mtp_tokens", "draft_mtp_accepted", "stretch_passes", "stretch_discarded")}
run(False, True)  # warm-up
outs = {}
for rnd in range(3):
    for label, use, pipe in (("off", False, True), ("old", True, False), ("new", True, True)):
        out, rate, c = run(use, pipe)
        outs.setdefault(label, out)
        print(f"round {rnd} {label:4s}: {len(out)} tok {rate:6.1f} tok/s {c}", flush=True)
print("same tokens off/old/new:", outs["off"] == outs["old"], outs["off"] == outs["new"])
