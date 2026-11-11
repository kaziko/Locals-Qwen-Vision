"""Issue #22: the cost of a decode pass of 1, 2 and 3 tokens, and where it goes.
The model is loaded with Locals' Runtime (float16, fused layers, small-batch kernels), as mtp_test.py loads it. The
cache holds a prompt and its greedy reply (at most 192 tokens), as in mtp_test.py's timing. Then, in each of
--rounds rounds, the passes of S = 1, 2 and 3 tokens in the round's order: the order rotates by one each round
(1 2 3, then 2 3 1, then 3 1 2) and each round's sync measurements run it forward and back (1 2 3 3 2 1), so each
pass is timed twice a round, and a drift within the round reaches every pass alike. With a multiple of 3 rounds
(the default 6), each pass goes first equally often:
  sync      mtp_test.py's timing (a fresh cache, 3 warm steps, 30 timed steps, one mx.eval a step); per round, the
            mean of a pass's two measurements, the same for every pass
  split     the same steps, each cut into building the graph (Python), encoding it (mx.async_eval returns) and
            waiting for the GPU (mx.eval returns); medians
  pipelined step s+1 built and queued while step s runs, as the engines decode (the first one queued before the
            timer starts): the GPU's time, with the CPU's time hidden behind it
  probe     right before each sync measurement's timed steps (after its fresh cache and warm steps) and right after
            them, 30 synchronous calls of MLX's own 4-bit matmul (mx.quantized_matmul, no Locals code) on the
            output head's weights, 1 row: how fast this GPU is at that moment
While it runs, other Locals processes see the GPU as busy (timing_setup.BusyMarker: a server's idle tuner waits);
the result's setup holds the scripts' hashes and the Locals servers running at the start and at the end.
First it prints the kernel each 4-bit matmul shape gets at 1, 2 and 3 rows.
--force M=FAMILY gives every shape at M rows that kernel family's tuned winner (mlx = MLX's own kernel);
--force M@NxK=FAMILY only the shape N x K. Repeatable. The choices are set in this process only. A spec that
cannot be read, or that forces no shape, stops the run.
Usage: step_timing.py MODEL OUT.json [--rounds 6] [--force SPEC ...]"""
import argparse
import json
import os
import platform
import statistics as stt
import subprocess
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

import locals as L
from locals import kernels as K
from locals.runtime import Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import timing_setup as TS  # noqa: E402  (docs/measurements/timing_setup.py)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--force", action="append", default=[])
a = ap.parse_args()
forces = []  # (spec, rows, "NxK" or "", family) of each --force, checked before the model loads
for spec in a.force:
    rows, eq, fam = spec.partition("=")
    M, _, only = rows.partition("@")
    if not (eq and fam and M.isdigit() and int(M) > 0):
        raise SystemExit(f"--force {spec}: expected M=FAMILY or M@NxK=FAMILY")
    forces.append((spec, int(M), only, fam))
busy = TS.BusyMarker().start()

repo = os.path.dirname(os.path.dirname(L.__file__))
commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "locals"], capture_output=True, text=True,
                       check=False).stdout.strip()
dev = mx.device_info()
setup = {"mac": f"{dev['device_name']} ({dev['architecture']}, {dev['memory_size'] >> 30} GB)", "macos": platform.mac_ver()[0],
         "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "commit": commit + (" with changes in locals/" if dirty else ""),
         "model": a.model, "tune_cache": K._TUNE_PATH, "preset": K._preset_path(), "force": a.force, "method": "issue-25",
         "scripts": TS.script_hashes(__file__, os.path.join(HERE, "summarize.py"), TS.__file__),
         "servers_at_start": TS.server_pids()}
for k, v in setup.items():
    print(f"{k}: {v}", flush=True)

rt = Runtime.load(a.model)
model, tok = rt.model, rt.tokenizer
tune = K._load_tune()
# (issue #31: each shape records its MLX version, so the file has no one version; results before it hold tune_cache_mlx)
setup["tune_cache_stale"] = len(K._stale_keys(tune))
print(f"tuning file has {setup['tune_cache_stale']} shapes timed under another MLX (stale)", flush=True)

# (N, K, dtype name) -> the layers. The dtype is the one warm-up and K._choose key a choice on (K._mod_dt: the
# activations' dtype, Runtime's compute dtype)
shapes = {}
for name, mod in model.named_modules():
    if K._tunable(mod):
        shapes.setdefault((mod["weight"].shape[0], mod["weight"].shape[1] * 8, K._mod_dt(mod)), []).append(name)

for spec, M, only, fam in forces:
    hit = [(N, Kd, dt) for N, Kd, dt in shapes if not only or only == f"{N}x{Kd}"]
    if not hit:
        raise SystemExit(f"--force {spec}: no 4-bit matmul shape {only}; the shapes are "
                         + ", ".join(sorted({f"{N}x{Kd}" for N, Kd, _ in shapes})))
    for N, Kd, dt in hit:
        c = ("mlx",) if fam == "mlx" else tune.get(K._tune_key(N, Kd, M, dt) + "|" + fam)
        if c is None:
            raise SystemExit(f"--force {spec}: no tuned {fam} kernel for {K._tune_key(N, Kd, M, dt)}")
        K._CHOICE[(N, Kd, M, dt)] = tuple(c)

# the kernel each matmul shape has at 1, 2 and 3 rows. Runtime.load's warm-up has already chosen them (K._CHOICE) for
# 1 to 16 rows, so the table shows each shape's choice, not that a pass used it; the passes below ran in the runs and
# stay. None: no choice recorded
x0 = mx.array([[100, 200, 300]])
for S in (1, 2, 3):
    mx.eval(model(x0[:, :S], cache=make_prompt_cache(model)))
choices = {}
print("\nkernel per matmul shape (layers using it): 1 row | 2 rows | 3 rows", flush=True)
for (N, Kd, dt), names in sorted(shapes.items()):
    row = [K._CHOICE.get((N, Kd, M, dt)) for M in (1, 2, 3)]
    choices[f"{N}x{Kd}"] = {"layers": len(names), "example": names[0], **{str(M): c for M, c in zip((1, 2, 3), row)}}
    print(f"  {N}x{Kd} ({len(names)}, e.g. {names[0].removeprefix('language_model.model.')}): "
          + " | ".join(str(c) for c in row), flush=True)

eos = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain how a refrigerator works, in about 150 words."}],
                                            add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
cache = make_prompt_cache(model)
logits, reply = model(mx.array(prompt)[None], cache=cache), []
for _ in range(192):
    y = int(mx.argmax(logits[0, -1]).item())
    reply.append(y)
    if y in eos:
        break
    logits = model(mx.array([[y]]), cache=cache)
ids = mx.array(prompt + reply)[None]
print(f"\ncache: {ids.shape[1]} tokens (prompt {len(prompt)}, reply {len(reply)})", flush=True)
tokx = int(ids[0, 5].item())  # the token mtp_test.py's timing repeats (feats[0][1][0, 5]), so the two time the same steps


def filled(S):
    c = make_prompt_cache(model)
    mx.eval(model(ids, cache=c))
    x = mx.array([[tokx] * S])
    for _ in range(3):
        mx.eval(model(x, cache=c))
    return c, x


head = K.head_module(K.text_model(model))
xp = mx.random.normal((1, head["weight"].shape[1] * 8), key=mx.random.key(0)).astype(mx.float16)


def probe(n=30):
    def f():
        return mx.quantized_matmul(xp, head["weight"], head["scales"], head["biases"], transpose=True, group_size=64, bits=4)
    for _ in range(3):
        mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


def sync(S, n=30):
    """(ms a step, the probe right before the timed steps, the probe right after them)"""
    c, x = filled(S)
    with busy.quiet():
        before = probe()
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(model(x, cache=c))
        ms = (time.perf_counter() - t) / n * 1e3
        return ms, before, probe()


def split(S, n=30):
    c, x = filled(S)
    parts = [[], [], []]
    with busy.quiet():
        for _ in range(n):
            t0 = time.perf_counter()
            y = model(x, cache=c)
            t1 = time.perf_counter()
            mx.async_eval(y)
            t2 = time.perf_counter()
            mx.eval(y)
            t3 = time.perf_counter()
            for p, v in zip(parts, (t1 - t0, t2 - t1, t3 - t2)):
                p.append(v * 1e3)
    return [stt.median(p) for p in parts]


def pipelined(S, n=30):
    """n steps, each built and queued while the one before runs; the first is queued before the timer starts, so
    no build is left in the open"""
    c, x = filled(S)
    mx.eval(model(x, cache=c))
    with busy.quiet():
        y = model(x, cache=c)
        mx.async_eval(y)
        t = time.perf_counter()
        for _ in range(n - 1):
            y2 = model(x, cache=c)
            mx.async_eval(y2)
            mx.eval(y)
            y = y2
        mx.eval(y)
        return (time.perf_counter() - t) / n * 1e3


sync(1)  # (warm)
PASSES = (1, 2, 3)
rounds = []
t_start = time.perf_counter()
for r in range(a.rounds):
    t_round = time.perf_counter() - t_start
    order = PASSES[r % 3:] + PASSES[:r % 3]  # 1 2 3, 2 3 1, 3 1 2, ...
    sy = []
    for S in order + order[::-1]:
        ms, before, after = sync(S)
        sy.append({"S": S, "ms": ms, "probe_before": before, "probe_after": after})
    sp = {str(S): split(S) for S in order}
    pp = {str(S): pipelined(S) for S in order}
    rounds.append({"t_s": t_round, "order": list(order), "sync": sy, "split": sp, "pipelined": pp})
    print(f"round {r} (t={t_round:.0f} s, order {' '.join(map(str, order))}): sync "
          + ", ".join(f"S={m['S']} {m['ms']:.2f}" for m in sy)
          + " ms | probe before, after each " + ", ".join(f"{m['probe_before']:.3f} {m['probe_after']:.3f}" for m in sy)
          + " ms | split (build + encode + wait) "
          + ", ".join(f"S={S} {b:.2f}+{e:.2f}+{w:.2f}" for S, (b, e, w) in sp.items())
          + " | pipelined " + ", ".join(f"S={S} {v:.2f}" for S, v in pp.items()), flush=True)
TS.end_of_run(busy, setup)


def summary(vals):
    return {"mean": stt.mean(vals), "min": min(vals), "max": max(vals)}


def per_round(mode, S):
    """S's value in each round: sync, the mean of its two measurements; pipelined, its one"""
    if mode == "pipelined":
        return [r["pipelined"][str(S)] for r in rounds]
    return [stt.mean(m["ms"] for m in r["sync"] if m["S"] == S) for r in rounds]


out = {"setup": setup, "cache_tokens": int(ids.shape[1]), "choices": choices, "rounds": rounds, "summary": {}}
print(f"\nover {a.rounds} rounds: mean (min to max), ms; ratio = mean / mean of S=1; sync: per round, the mean of "
      f"each pass's two measurements", flush=True)
for mode in ("sync", "pipelined"):
    base = stt.mean(per_round(mode, 1))
    for S in PASSES:
        sm = summary(per_round(mode, S))
        sm["ratio"] = sm["mean"] / base
        out["summary"][f"{mode}_S{S}"] = sm
        print(f"  {mode:9s} S={S}: {sm['mean']:.2f} ({sm['min']:.2f} to {sm['max']:.2f}), {sm['ratio']:.2f}x", flush=True)
pv = [m[k] for r in rounds for m in r["sync"] for k in ("probe_before", "probe_after")]
pm = out["summary"]["probe"] = summary(pv)
print(f"  probe (before and after each sync measurement): {pm['mean']:.3f} ({pm['min']:.3f} to {pm['max']:.3f}) ms",
      flush=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
