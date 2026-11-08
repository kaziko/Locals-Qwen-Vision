"""The tables of README.md from the result files of step_timing.py: each run's means, and how the slow sync
measurements line up with the probe (MLX's own matmul). It reads both methods' files: the old one (issue #22: the
probe before each measurement's fresh cache, S = 1 timed twice a round) and the new one (issue #25: the probe right
before and right after the timed steps, the slower of the two counting here, every pass timed twice a round).
Usage: summarize.py results/*.json"""
import json
import os
import statistics as stt
import sys
from collections import namedtuple

SLOW_PROBE = 1.25  # a probe this many times its median marks a slow moment (a line chosen here, not given)
SLOW_STEP = 1.10  # a step this many times its run's median counts as slow (the same)
Row = namedtuple("Row", "run step probe ratio")  # ratio: the step's time / that step's median in its run


def spread(sm, f=".2f"):
    return f"{sm['mean']:{f}} ({sm['min']:{f}} to {sm['max']:{f}})"


runs = {}
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    rounds = d.get("rounds")
    if rounds == []:
        print(f"(skipped {os.path.basename(path)}: no rounds)")
    elif rounds and ("probe_before_sync" in rounds[0] or isinstance(rounds[0].get("sync"), list)):
        runs[os.path.basename(path)[:-5]] = d
    else:
        print(f"(skipped {os.path.basename(path)}: not a step_timing.py result with the probe)")
if not runs:
    sys.exit("no step_timing.py result with the probe and at least one round")

print("run | rounds | force | sync S=1, S=2, S=3: mean (min to max) ms, S=2 / S=1 | pipelined S=1, S=2, S=3, S=2 / S=1 | probe")
for name, d in runs.items():
    s = d["summary"]
    sync = ", ".join(spread(s[f"sync_S{S}"]) for S in (1, 2, 3))
    pipe = ", ".join(f"{s[f'pipelined_S{S}']['mean']:.2f}" for S in (1, 2, 3))
    print(f"{name} | {len(d['rounds'])} | {' '.join(d['setup']['force']) or '-'} | {sync}, {s['sync_S2']['ratio']:.2f}x | "
          f"{pipe}, {s['pipelined_S2']['ratio']:.2f}x | {spread(s['probe'], '.3f')}")

print("\nrun | split, the mean over rounds of each round's medians: build + encode + wait (ms) for S=1, S=2, S=3")
for name, d in runs.items():
    parts = {S: [stt.mean(r["split"][S][i] for r in d["rounds"]) for i in range(3)] for S in ("1", "2", "3")}
    print(f"{name} | " + ", ".join(f"S={S} {b:.2f} + {e:.2f} + {w:.2f}" for S, (b, e, w) in parts.items()))

def measurements(d):
    """[(step, ms, probe ms)] of a run's sync measurements, step by step"""
    rounds = d["rounds"]
    if "probe_before_sync" in rounds[0]:  # (the old method)
        return [(k, r["sync"][k], r["probe_before_sync"][k]) for k in ("1_first", "2", "3", "1_last") for r in rounds]
    return [(str(m["S"]), m["ms"], max(m["probe_before"], m["probe_after"])) for r in rounds for m in r["sync"]]


rows = []
for name, d in runs.items():
    ms = measurements(d)
    for step in dict.fromkeys(k for k, _, _ in ms):
        med = stt.median(v for k, v, _ in ms if k == step)
        rows += [Row(name, step, p, v / med) for k, v, p in ms if k == step]
pm = stt.median(x.probe for x in rows)


def slow_probe(x):
    return x.probe > SLOW_PROBE * pm


print(f"\n{len(rows)} sync measurements in {len(runs)} runs; the probe's median {pm:.3f} ms")
for label, group in ((f"probe more than {SLOW_PROBE}x its median", [x for x in rows if slow_probe(x)]),
                     (f"probe at most {SLOW_PROBE}x its median", [x for x in rows if not slow_probe(x)])):
    rel = [x.ratio for x in group]
    avg = f", step / its run's median {stt.mean(rel):.3f} on average" if rel else ""  # (no mean of an empty group)
    print(f"  {label}: {len(group)} measurements{avg}, {sum(r > SLOW_STEP for r in rel)} of them more than {SLOW_STEP}x")
print(f"measurements more than {SLOW_STEP}x their run's median (run, step, probe ms, ratio; * the probe was slow too):")
for x in sorted(rows, key=lambda x: -x.ratio):
    if x.ratio > SLOW_STEP:
        print(f"  {x.run}, S={x.step}, probe {x.probe:.3f}{' *' if slow_probe(x) else ''}, {x.ratio:.2f}x")
