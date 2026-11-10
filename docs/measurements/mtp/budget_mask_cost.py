"""Issue #39 item 2: the cost of the budget masks in a constrained pass of the pipelined loop, far from max_tokens.
One greedy JSON reply under a schema, alone, MTP drafts pipelined; the time inside the stretches over their passes,
and how many budget masks the stretches built with a device token count (before the gate: two a pass; after it:
none until the budget can bite). A warm-up run, then RUNS timed runs; prints each and their median.
Run from the repo root: HF_HOME=... LOCALS_DISK_CACHE=0 .venv/bin/python docs/measurements/mtp/budget_mask_cost.py <model> <mtp file>
"""
import statistics as st
import subprocess
import sys
import time

import mlx.core as mx

from locals.engine import LocalsController
from locals.hybrid_batch import HybridBatchEngine
from locals.mtp import MTP
from locals.packed import Request
from locals.runtime import Runtime

RUNS = 5
SCHEMA = {"type": "object", "properties": {"books": {"type": "array", "items": {
    "type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer"},
                                     "genres": {"type": "array", "items": {"type": "string"}}, "summary": {"type": "string"}},
    "required": ["title", "author", "year", "genres", "summary"]}}}, "required": ["books"]}
PROMPT = ("Return a JSON object with a list of six fictional books: title, author, year, genres (a list) and a "
          "two-sentence summary for each.")
MAX_TOKENS = 700

model, mtp_path = sys.argv[1], sys.argv[2]
print("commit", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "| dirty files:", subprocess.run(["git", "status", "--porcelain", "locals"], capture_output=True, text=True).stdout.split())
rt = Runtime.load(model)
mtp = MTP.load(rt.model, mtp_path, drafts=1)
c = rt.compile(SCHEMA)
bt = LocalsController(c.constraint, c.planner, True, True).budget_tables()[0]
print(f"budget tables: max_cost {bt.max_cost}, has_inf {bt.has_inf}")

dev_masks = [0]
orig_mask = LocalsController.budget_mask


def counted(self, qg, r):
    if isinstance(r, mx.array):
        dev_masks[0] += 1
    return orig_mask(self, qg, r)


LocalsController.budget_mask = counted
in_stretch = [0.0]
orig_stretch = HybridBatchEngine._solo_stretch


def timed(self, *a, **k):
    t = time.perf_counter()
    try:
        return orig_stretch(self, *a, **k)
    finally:
        in_stretch[0] += time.perf_counter() - t


HybridBatchEngine._solo_stretch = timed
prompt = rt.encode_chat(PROMPT)


def run():
    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4, share_prefix=False, mtp=mtp)
    r = Request(list(prompt), LocalsController(c.constraint, c.planner, True, True), MAX_TOKENS)
    w = [r]
    dev_masks[0], in_stretch[0] = 0, 0.0
    t = time.perf_counter()
    eng.serve(lambda b: [w.pop()] if w else [])
    dt = time.perf_counter() - t
    n = eng.counters.get("stretch_passes", 0)
    return {"tokens": len(r.out), "finish": r.finish, "passes": n, "dev_masks": dev_masks[0],
            "ms_per_pass": 1000 * in_stretch[0] / max(n, 1), "tok_s": len(r.out) / dt, "out": r.out}


run()  # warm-up
rows = [run() for _ in range(RUNS)]
for x in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k != "out"})
print("same tokens in every run:", all(x["out"] == rows[0]["out"] for x in rows))
print(f"median ms/pass {st.median(x['ms_per_pass'] for x in rows):.3f}, median tok/s {st.median(x['tok_s'] for x in rows):.1f}")
