# MTP drafts: Qwen3.5's own draft layer

The measurements behind [issue #18](https://github.com/kaziko/Locals-Qwen-Vision/issues/18). Run on an M2 Max
(30-core GPU, 32 GB), 2026-11-06, at commit `f31121c` (its `locals/` is the same as `0e40a3b`'s), with the repo's
`.venv`: mlx 0.32.3, mlx-lm 0.32.0. This Mac's kernel choices were timed on mlx 0.32.4.

## What it measures

Qwen3.5-4B's checkpoint has one multi-token prediction (MTP) layer: from the main model's hidden state at
position i and the embedding of token i+1, it predicts token i+2. The 4-bit MLX files
(`mlx-community/Qwen3.5-4B-MLX-4bit` and `Kaziko/Qwen3.5-4B-Zwyrol`) do not have it.

- `download_mtp.py` fetches only the layer's 15 `mtp.*` tensors (241.2 MB, BF16) from `Qwen/Qwen3.5-4B`, with
  HTTP range requests, and saves them as one safetensors file.
- `mtp_test.py` loads the main model with `Runtime.load` (float16, fused layers, small-batch kernels, as the
  server runs it on this Mac) and writes a greedy reply, thinking off, at most 192 tokens, to each of 12 prompts.
  Then, teacher-forced over prompt + reply, it counts how often the layer's guess equals the main model's
  greedy token. It prints first that the main model, teacher-forced, gives its own greedy reply back.
- It scores 8 wirings: norm weights used as w or as 1+w, the hidden state before or after the final norm, and
  the order of the concatenation. For the best one, it runs a float16 and a 4-bit layer, and a second guess
  (the layer fed its own output again over the same sequence: an approximation of a chained draft).
- Last, it times a main step of 1, 2 and 3 tokens and one guess (the layer, the output head, argmax), with one
  `mx.eval` per step and no pipelining. The 1-token step is timed first and last, as the noise floor.
  The committed results come from that method, which took the lower of the two 1-token values. Since issue #25 each
  main step is timed twice, forward and back (1 2 3 3 2 1), and the mean of the two counts. While the script runs,
  other Locals processes see the GPU as busy. The result's `setup` holds the scripts' hashes and the Locals servers
  found at the start and at the end (see `../two-token-pass/README.md`). A first run with the new method, on
  Zwyrol at commit `936c6c0`, is in `results/method25/`: main steps 11.05, 12.80 and 16.35 ms for 1, 2 and 3 tokens
  (the mean of each pass's two), and no Locals server at the start or at the end. The command was the Zwyrol line
  below, with the output in `<scratch>/runs/mtp_method25_1.json` and `LOCALS_TUNE_CACHE` set to the copy that
  `../two-token-pass/README.md` describes. A second run, at commit `d0cc5b9` on 2026-11-08, right after the three
  full `step_timing.py` runs ("Three full runs of the new method" there, with the same setup), is
  `results/method25/full_1`: main steps 10.10, 12.45 and 15.81 ms (2 / 1: 1.23x). Its two 2-token values were 11.91
  and 13.00 ms, and this script has no probe, so its 2 / 1 is less sure than `step_timing.py`'s. The 4-bit MTP guess
  took 1.58 ms, 0.16 of a main step: estimate 1.32x with 1 draft, 1.30x with 2 (with 4-bit acceptance, before the
  pipelining loss).

## Commands

Run from this folder, one model at a time (never two GPU jobs at once):

```
<repo>/.venv/bin/python download_mtp.py mtp_bf16.safetensors       # 241 MB; *.safetensors is ignored here
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_test.py mtp_bf16.safetensors results/zwyrol.json Kaziko/Qwen3.5-4B-Zwyrol > results/zwyrol.txt 2>&1
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_test.py mtp_bf16.safetensors results/original.json mlx-community/Qwen3.5-4B-MLX-4bit > results/original.txt 2>&1
```

`<repo>` is the clone's folder; in the result files it stands for the path the run printed.

## Results

| | Zwyrol (`zwyrol.txt`) | Original (`original.txt`) |
|---|---|---|
| Main model gives its own greedy reply back | 1,633 of 1,635 | 1,685 of 1,690 |
| Best wiring | norms 1+w, hidden after the final norm, `[embedding, hidden]` | the same |
| First guess right, float16 layer | 83.7% (1,358 / 1,623) | 85.7% (1,438 / 1,678) |
| First guess right, 4-bit layer | 82.7% (1,343 / 1,623) | 85.4% (1,433 / 1,678) |
| Second guess right, given the first (4-bit) | 73.5% (979 / 1,332) | 74.8% (1,063 / 1,421) |
| The same, with the hidden state before the final norm (float16) | 82.0% | 84.5% |
| The six other wirings (float16) | 0.0% to 0.2% | 0.0% to 0.5% |
| Main step, 1 token (first, last) | 10.20, 10.14 ms | 10.47, 10.06 ms |
| Main step, 2 tokens | 14.05 ms | 13.53 ms |
| Main step, 3 tokens | 15.91 ms | 15.53 ms |
| One guess, float16 / 4-bit layer | 2.02 / 1.57 ms | 2.02 / 1.58 ms |

The by-prompt rates are in each result file. `results/original_10_prompts.*` is the first run, made with an earlier
version of `mtp_test.py`: the original model only, 10 prompts (without the two Polish tool-call prompts), and
the draft logits from the tied embedding (the script now takes the model's output head, Zwyrol's own
`lm_head`). In that run: first guess right 85.2% (float16) and 85.0% (4-bit); steps of 1, 2 and 3 tokens
10.35, 12.57 and 15.65 ms.

The 2-token step moved from run to run (1.23x to 1.39x of a 1-token step over the three runs) while the 3-token
step stayed at 1.53x to 1.57x. The "estimate" lines at the end of each result file use that run's own times; issue #18
has the estimate table and its assumptions.

## Gap test: the layer's KV with positions missing

`mtp_gap_test.py` asks how much the hit rate drops when the layer's own KV lacks positions: after a prompt-cache
hit (the main model's cache comes back, the layer's KV does not) and after steps the layer did not run. Same Mac
and `.venv`, 2026-11-06, at commit `6c972cc` (`locals/` the same as `f31121c`'s), Zwyrol, the best wiring, 4-bit
layer. The server on port 8321 was running beside it, idle; this test times nothing.

Teacher-forced over prompt + reply, as above. A missing key position is masked out of the layer's attention (each
query still sees itself), and the real positions are kept. 14 prompts: the 12 above, and two long ones, the first
5,000 characters of `locals/lookup.py` and the first 6,000 of `docs/engine/README.md` at that commit (the prompt
changes when those files change).

```
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_gap_test.py mtp_bf16.safetensors results/gap_zwyrol.json Kaziko/Qwen3.5-4B-Zwyrol > results/gap_zwyrol.txt 2>&1
```

First draft right (`results/gap_zwyrol.txt`):

| What the layer sees | All reply tokens | First 16 reply tokens | Reply tokens 48 on |
|---|---|---|---|
| Every earlier position (`full`) | 81.8% (1,640 / 2,005) | 87.1% (195 / 224) | 80.8% (1,106 / 1,369) |
| No prompt position before the last one (`no-prompt`) | 76.6% (1,535 / 2,005) | 73.2% (164 / 224) | 77.3% (1,058 / 1,369) |
| No reply position 16 to 47 (`gap32`) | 81.1% (1,626 / 2,005) | 87.1% (195 / 224) | 80.6% (1,104 / 1,369) |

- Checks: `full` with a boolean mask gives the same counts as with the `causal` mask (1,640 / 2,005), and the 12
  prompts of `mtp_test.py` give 1,343 / 1,623 again, as in `results/zwyrol.txt`.
- By group, from `results/gap_zwyrol.json`: the 3 tool calls 96.3% (104 / 108) with every position and 76.9%
  (83 / 108) without the prompt; the 2 long prompts 77.7% (297 / 382) and 70.2% (268 / 382).
- The tool-call replies (31 to 40 tokens) end before token 48, so `gap32` says nothing about them.

## Step 4: MTP drafts in the server, one request

Issue #18's step 4, with the engine at commit `f7efec9` (MTP drafts behind `LOCALS_MTP`, `locals/mtp.py`). Same M2 Max
(30-core GPU, 32 GB), mlx 0.32.3, mlx-lm 0.32.0, Zwyrol, 2026-11-07. `step4.sh` starts a server of its own from this
checkout on port 8399 (`LOCALS_DISK_CACHE=0`), checks that its `/health` build is the checkout's (`0fd1cbbd8ed2` in
every run), runs the decode sweep of `docs/measurements/m2-max.md`'s appendix at 1 request (`decode_sweep.py`, with
the port as a parameter: 256 tokens, end of text ignored, greedy, a warm-up and 2 rounds) and
`locals bench --agents --n 1` (a warm-up and 2 rounds, the server's temperature 0.2), saves `/stats`, and stops only its
own server. Nothing else ran on the GPU. The unchanged server (MTP off) ran twice first, as the noise floor.

```
bash step4.sh off-a && bash step4.sh off-b && bash step4.sh on1 1 && bash step4.sh on2 2
```

Each round on its own line (`results/step4/<label>_sweep.txt` and `_agents.txt`):

| Run | Decode sweep, 1 request | `bench --agents --n 1` (41 tokens, 11 forced) |
|---|---|---|
| MTP off (`off-a`) | 110.2, 110.1 tok/s | 51, 52 tok/s |
| MTP off (`off-b`) | 110.8, 109.7 tok/s | 51, 52 tok/s |
| MTP on, 1 draft (`on1`) | 108.6, 108.2 tok/s | 54, 51 tok/s |
| MTP on, 2 drafts (`on2`) | 107.5, 109.3 tok/s | 50, 50 tok/s |

Drafts over each whole run (warm-ups included), from `results/step4/<label>_stats.json`:

| Run | MTP drafts kept | Prompt-lookup drafts kept | Steps |
|---|---|---|---|
| `off-a` | | 33 of 85 | 841 |
| `on1` | 293 of 348 (84.2%) | 39 of 72 | 536 |
| `on2` | 360 of 567 (63.5%) | 36 of 93 | 474 |

- Live, the first draft is kept as often as the teacher-forced rate above says (84.2% against 82.7%), and the steps
  fall from 841 to 536. The time does not: a `_solo` pass reads its draft back to the CPU before it runs and its
  picks after, so each pass waits for the CPU twice, and the pipelining of plain steps is lost. The issue's estimate
  for this case ("full output head, each step waits for the CPU") was 1.00x for 1 draft; measured, 0.98x.
- Greedy replies (`greedy_check.py`, in process, the 14 prompts of `tests/test_mtp_model.py`, at most 192 tokens):
  11 are the same with MTP drafts, 3 are not (`results/greedy_check.txt`). Each of the 3 first differs where the two
  tokens' logits, teacher-forced, are equal or one float16 step apart (24.062 / 24.078; 30.297 / 30.297; 23.641 /
  23.625): a pass over several tokens rounds differently from a one-token step. In that run the MTP drafts were kept
  629 of 755 times (83.3%) with 1 draft, 844 of 1,197 (70.5%) with 2.

## Issue #37: the pipelined loop, one draft

The build of `docs/adr/0002-pipelined-draft-loop.md` (`_solo_stream` in `locals/hybrid_batch.py`): with 1 draft a
pass, a pass is launched before the last one is read. Same M2 Max (30-core GPU, 32 GB), mlx 0.32.3, mlx-lm 0.32.0,
Zwyrol (from a Hugging Face cache on another disk, `HF_HOME`), `step4.sh` as above with the labels below, the tuning
file read from a copy (`LOCALS_TUNE_CACHE`), 2026-11-10. Other programs ran on this Mac meanwhile (a 3D scene app,
Time Machine, an agent process), none of them a Locals server; the noise floor below is theirs too. Three sessions, each
with the unchanged server (MTP off) run twice as the noise floor:

- `pre37-*`: before, commit `1d8d629` (build `92a06e769386`), the loop that reads every pass.
- `post37-*`: the first build of this change (build `581d32502c07`), which selected each recurrent layer's state
  with `where` and ran its bookkeeping as some thirty small ops; the whole Mac was slower in that session (MTP off at
  104 to 106 tok/s against 110 to 114 in the other two).
- `post37b-*`: the build committed with this section (`locals/` as in the commit that adds it; build `01c09b7ead35`),
  with the state selected by two gathers a layer and the bookkeeping compiled (`_stream_book`): MTP off, then the
  pipelined loop with the back-off after rejected drafts (the ADR's design, `LOCALS_MTP_BACKOFF=1`, the default), then
  without it (`LOCALS_MTP_BACKOFF=0`), in that order twice.

Each round on its own line (`results/step4/<label>_sweep.txt` and `_agents.txt`):

| Run | Decode sweep, 1 request | `bench --agents --n 1` (41 tokens, 11 forced) |
|---|---|---|
| Before, MTP off (`pre37-off-a`, `-b`) | 112.4, 113.3; 110.5, 111.5 tok/s | 50, 52; 51, 52 tok/s |
| Before, MTP on, 1 draft (`pre37-on1`) | 119.1, 109.9 tok/s | 51, 54 tok/s |
| First build, MTP off (`post37-off-a`, `-b`) | 104.3, 104.6; 105.9, 104.7 tok/s | 48, 49; 48, 49 tok/s |
| First build, pipelined (`post37-on1-a`, `-b`) | 110.0, 109.8; 111.6, 112.3 tok/s | 45, 45; 48, 46 tok/s |
| Committed build, MTP off (`post37b-off-a`, `-b`) | 113.0, 113.7; 109.6, 111.1 tok/s | 51, 53; 50, 53 tok/s |
| Committed build, pipelined, back-off on (`post37b-on1-a`, `-b`) | 128.6, 129.3; 126.0, 126.3 tok/s | 51, 51; 50, 51 tok/s |
| Committed build, pipelined, back-off off (`post37b-nb-a`, `-b`) | 144.2, 143.7; 142.8, 141.7 tok/s | 53, 52; 52, 52 tok/s |

Against the MTP-off runs of its own session (109.6 to 113.7 tok/s, a 4 tok/s spread), the pipelined loop is 1.14x with
the back-off (126.0 to 129.3) and 1.28x without it (141.7 to 144.2); the estimate in #29 was about 1.35x, the loop that
reads every pass 0.98x (step 4 above) and 1.03x here. The agents bench does not move: its reply is 41 tokens with 11
forced, and each forced span ends a stretch (one pass discarded, the pipeline drained).

Drafts over each whole run (warm-ups included), from `results/step4/<label>_stats.json`:

| Run | MTP drafts kept | Cut to nothing | Prompt-lookup drafts kept | Steps (stretch passes, discarded) |
|---|---|---|---|---|
| `pre37-on1` | 293 of 348 (84.2%) | | 39 of 72 | 536 |
| `post37b-on1-a` | 321 of 378 (84.9%) | 3 | 12 of 27 | 546 (519, 12) |
| `post37b-nb-a` | 372 of 452 (82.3%) | 3 | 12 of 24 | 494 (467, 12) |

(`post37-on1-a` and the `-b` runs have the same counts as their `-a`: greedy, the same prompts.) Prompt-lookup drafts
are not consulted inside a stretch; the ones here ran at boundaries, before a stretch started.

**Why the back-off costs a tenth.** In the loop that reads every pass, a draft pass costs the pipelining of the main
loop, so after a rejected draft the row ran plain steps for 1, 2, 4, ... 32 tokens (`solo_miss`, `solo_resume`), and
one more each time: the main loop reads a step while the next runs, so it commits one token past `solo_resume` before
the boundary that brings the row back to `_solo` (the pipelined loop matches that, so the seeded regression test holds
token for token). In the pipelined loop every pass feeds two tokens whether it uses the draft or not, so a back-off
pass costs what a draft pass costs and commits one token for sure instead of 1.85 on average: 519 passes for the same
tokens as 467 without the back-off. The back-off stays the default here, as the ADR has it; `LOCALS_MTP_BACKOFF=0`
turns it off in both loops.

**Where a pass's time goes** (`stream_profile.py`, `stream_ablation.py` and `pass_cost.py`, in process, greedy, the
`code` prompt of `tests/test_mtp_model.py`, back-off on). On the committed build (`results/stream37/stream_profile.txt`,
`pass_cost.txt`): a pass of the pipelined loop spends 2.5 ms building its graph (Python), then 9.8 ms in
`mx.async_eval`, then 0.00 ms waiting for the pass before it (12.6 ms a pass in all, 119 tok/s there); a plain step
2.2 and 6.5 ms (104 tok/s). `mx.async_eval` does not return while the pass before is still running on the GPU: with
nothing in flight (`pass_cost.py`: one pass, then `mx.eval`) it encodes a 1-token step in 1.4 to 1.5 ms, after which the
GPU takes 7.6 to 7.7 ms, and a 2-token pass in 1.4 ms with 9.4 ms of GPU (9.0 to 10.0 with the recurrent layers'
per-token states recorded). So both loops are bound by the GPU, and a pass of the pipelined loop costs it about 11 ms:
the 2-token pass, the states, the MTP layer and its draft through the output head (1.6 ms, "one guess" above), the
state gathers, the sampling. The first build's `where` on each of the
24 recurrent layers' states and its thirty small bookkeeping ops took 1.7 ms more of encoding per pass
(`stream_ablation.py` on the first build, printed in that session: 11.1 ms in `mx.async_eval`, 9.4 ms with the
selection skipped, 10.65 ms with the MTP layer's fill skipped, the same with the per-token states not recorded), which
is what the committed build cut.

**Replies.** `greedy_check.py` on the committed build, back-off on (`results/greedy_check_37.txt`): with 1 draft (the
pipelined loop) 12 of the 14 greedy replies are the same as MTP off's and 2 first differ where the two tokens' logits
are equal (`en-explain` at token 19, `math` at 15: gap 0.0000); step 4 above had 11 of 14, with gaps up to one float16
step. Drafts kept 746 of 898 (83.1%). `tests/test_mtp_model.py` (the server, 14 prompts, a cached next turn, two
requests at once, an image prompt): with the back-off on 40 passed and 1 skipped, with it off 39 passed and 2 skipped (a
next turn whose turn before differed at a tie, as the test allows).

`seeded_check.py` (`results/stream37/seeded_real.txt`, the first 10 prompts, 192 tokens, temperature 0.7, seeds 1 and 2,
in process on the first build, back-off on): the greedy reply of the pipelined loop equals the loop that reads every
pass's on all 10; a seeded sampled reply equals it on 2 of 20 runs (`en-explain` seed 1, `tool-call` both seeds) and
first differs between token 21 and 170 on the other 18. On the tiny float32 model of `tests/test_mtp.py` the two loops
agree token for token in every case (5 draft patterns, 2 seeds, back-off on and off); on the real model a 2-token pass
rounds its float16 logits differently from the 1-token step the other loop runs in its back-off (the effect
`greedy_check` shows for greedy), and a sampled draw near a boundary flips on that.

`edit_workload.py` (`results/stream37/edit_workload.txt`, the repeated-module edit of `tests/test_solo.py`, greedy, 200
tokens, in process on the first build): MTP off, prompt-lookup drafts 108.7 to 116.8 tok/s (114 of 253 draft tokens
kept); the loop that reads every pass 94.4 to 96.2; the pipelined loop 119.5 to 120.7 (every one of its 100 drafts kept,
no lookup drafts inside the stretch). The same tokens in all three.
