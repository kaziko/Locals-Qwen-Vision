"""Usage: turn2_cost.py MODEL — a follow-up turn about two images: its time, its cached tokens, and its answer.
Each run empties the prompt cache, asks the first turn, then times the second."""
import statistics
import sys
import time

import locals as L
from locals.server import Server

import request

model_id = sys.argv[1]
print("locals from", L.__file__, flush=True)
first = request.messages("Describe both pictures in one sentence.")
srv = Server(model_id, engines=("locals",), preload=("locals",))
ts, cached, answers = [], [], []
for _ in range(4):
    request.empty_cache(srv)
    a = srv.complete({"model": "locals", "max_tokens": 40, "temperature": 0, "messages": first})["choices"][0]["message"]["content"]
    follow = first + [{"role": "assistant", "content": a}, {"role": "user", "content": "Which one is brighter?"}]
    t = time.perf_counter()
    r = srv.complete({"model": "locals", "max_tokens": 1, "temperature": 0, "messages": follow})
    ts.append(time.perf_counter() - t)
    cached.append(r["usage"]["prompt_tokens_details"]["cached_tokens"]); answers.append(r["choices"][0]["message"]["content"])
print(f"{model_id}: turn 2 of {r['usage']['prompt_tokens']} tokens, cached {cached[1:]}, median {statistics.median(ts[1:])*1000:.0f} ms "
      f"(runs 2-4: {', '.join(f'{t*1000:.0f}' for t in ts[1:])}), first token {answers[1:]!r}", flush=True)
srv.unload()
