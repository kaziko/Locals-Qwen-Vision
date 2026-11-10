---
status: accepted
---

# A draft pass is launched before the last one is read; a rejected draft slot stays as a hole

With `LOCALS_MTP` on and one request alone, each `_solo` pass (issue #29) reads two things back to the CPU: the
draft before the pass, the picks after it. The next pass waits for both. Measured, 1 draft runs at 0.98x of MTP
off (#18, step 4); the estimate with no wait is about 1.35x (#29, the comment of 2026-11-10). The plain loop
already launches step s+1 before it reads step s. The solo loop cannot, because what pass p+1 needs depends on
how many draft tokens pass p kept: the cache length (today a rewind), the next token, and the next draft. We
propose: with one draft, every one of those becomes a device value computed from pass p's logits, and the main
cache is never trimmed inside the stretch. A rejected draft slot stays in the KV as a **hole**: masked out of
attention, as the MTP entry already does with its flag column (`_Flagged`), and skipped by the positions.

## What moves to the GPU

| Today, on the CPU | Proposed, a device value |
|---|---|
| `kept`: the picks read back, compared to the draft | greedy: `argmax(logits[0]) == d`; sampled: `u < p(d)` (the test `_solo_verify` does now, unread) |
| the next token `x'` | `where(kept, pick[1], pick[0])`; sampled: the draws `_solo_verify` makes, unread |
| `_solo_rewind` of the recurrent layers | `token_states[kept]` by a device index (`take`), the conv state the same way |
| `_solo_rewind` of the attention layers (`trim`) | no trim: the slot's flag is `kept`; attention masks flagged-out slots |
| the rope offset (`cache.offset`, a Python int in `fuse.py`) | a device counter of real slots; `mx.fast.rope` takes `offset: int | array` (mlx 0.32.3, checked in its signature) |
| the next MTP draft (`MTP.draft`, `.tolist()`) | `fill` on the hidden state at the kept position with `x'`, its argmax fed as the draft, unread |
| the grammar state after the kept tokens | `_advance` is already a device table lookup; `_solo_masked` already builds the per-position masks on the device |

## What stays on the CPU, one pass behind

The CPU reads pass p's picks while pass p+1 runs, as the plain loop does, and runs `_commit_drip` on them:
end of sequence, stop strings (`controller.finished()`), `max_tokens`, forced spans, streaming (`on_token`).
When it finds one of these, pass p+1 was launched on a token that should not have been fed:

- **End of sequence, stop string, `max_tokens`:** the row is done; pass p+1 is wasted. The plain loop wastes one
  step the same way today.
- **A forced span** (`controller.forced(y)`): the span follows the committed token `y`, which pass p+1 did feed;
  its draft slot is marked as a hole, and the span is fed in the next pass as today.

A pass launched on a token that should not have been fed is a **discarded pass**: nothing of it is committed, and
it consumes no index of the random stream (below).

The draft checks `_draft_checked` makes before a pass today (an end-of-sequence draft, a draft the grammar does
not allow, the `max_tokens` budget) and the back-off after rejections (`solo_miss`, `solo_resume`) cannot run on
the CPU before the launch: the draft is a device value. They become one device flag, `valid`: the draft is not
an end-of-sequence token, the grammar allows it (`allowed[g][d']` from the grammar table), it is inside the
budget (a device count of committed tokens), and the back-off is over (device `miss` and `resume` counters). With `valid` false the pass still feeds
`[x', d']`, but position 0 is sampled by the draft-less rule and the slot `d'` is a hole: the same tokens and the
same draws as today's draft-less pass, at the cost of a 2-token pass where today runs a 1-token step.

The reply is the same token for token: only the main model's own picks are committed, as now.

## The random stream of a seeded request

A seeded request draws from keys `row_key(seed, n)`; `r.draws` is the index `n` of the next key. Today's
schedule, which the new loop keeps variate for variate:

| Pass | Key `n` is used for | `n` |
|---|---|---|
| a plain step, or a solo pass with no draft (`_specs`) | the Gumbel noise of the one sampled token | +1 |
| a solo pass with k draft tokens (`_solo_verify`) | split in two: one uniform per draft position (all k drawn at once), and exactly one categorical draw: the residual at the first rejected position, else the bonus token after the k kept | +1, whatever the outcome |
| greedy (temperature 0), a forced-span token, a pass not run (a draft cut to nothing, the back-off) | nothing | +0 |

In the pipelined loop the index is assigned at launch, before pass p's outcome is known; the key depends only
on the index, so that is safe. The pass computes both rules from that key, the draft rule (split key, uniform,
categorical) and the draft-less rule (Gumbel), and `valid` selects one with `where`; `kept` selects between the
residual and the bonus draw, both from the second half, as today's code draws exactly one of them. A discarded
pass gives its index back: the pass that replaces it (a forced span) uses the same `n`. A greedy request assigns
no index.

The proof is a seeded regression test: the same request through today's solo loop and the new one, token for
token, over replies with kept drafts, rejected drafts, a cut draft and a back-off. MTP off is not the
reference: the two rules draw different variates from the same key, so MTP on and off give different replies
for one seed today as well, with the same distribution.

## Consequences

- The main KV holds holes inside a solo stretch, one per rejected pass (about 16% of passes with 1 draft, from
  the 84.2% kept in #18). The stretch ends with a compaction: the real slots copied down and the counter made the
  offset, before another row joins, before a prompt-cache entry or a checkpoint is saved, and before the KV is
  read by anything that uses `cache.offset`. A rejected slot's position is also skipped by the positions, so the
  attention is over the same slots at the same positions as today.
- The fused attention path in `fuse.py` takes the offset from `cache.offset`; it must take the device counter
  for a one-row cache with holes. Everything else that reads `cache.offset` sees the cache only after compaction.
- 1 draft only. With 2 drafts `kept` is 0, 1 or 2, and the next draft's input position has three cases;
  `LOCALS_MTP_DRAFTS=2` keeps today's loop.
- The hole at a kept cache's end (#29 item 2) is a separate, small change: the last hidden state kept with the
  cache entry, and `_mtp_last` fed from it; it does not depend on this.
- Proof is end to end: the step-4 decode sweep (`docs/measurements/mtp/step4.sh`) before and after, on the same
  machine and build, with two unchanged runs as the noise floor; `greedy_check.py` for identity.

## Considered options

- **A cache offset that is a device value**, so a rejected slot is written over. Every cache class, the fused
  kernels and the prompt cache read `cache.offset` as a Python int; that is a rewrite of the cache, not of the
  loop.
- **Launch pass p+1 for both outcomes** (kept and rejected) and keep one. Two passes for one: the whole gain.
- **Keep the CPU in the loop, read less** (one flag instead of the picks). Still one sync per pass, so the
  pipelining of the stream is still lost; #18 puts that at 16% with one request. The 0.98x is mostly that.

## Decisions on the open points (PR #36, 2026-11-10)

1. **Holes in the main KV; `cache.offset` stays a Python int.** The physical offset counts stored slots; a
   device-side logical position counter, used only in the rope path of a one-row cache during a solo stretch,
   counts real positions. Compact before anything that expects a dense cache.
2. **Compact at the stretch end only.** The CPU catching up is not a boundary: the same stretch goes on. One
   gather at the boundary that needs a dense cache anyway; an adaptive threshold, if long stretches ever make the
   masked length costly, is a separate optimization.
3. **The random stream**, as the section above: the same variates from the same key indices, in the same
   order as today's solo loop; the reference for the seeded test is that loop, not MTP off.

## Outcome (issue #37, 2026-11-10)

Built as `_solo_stretch` in `locals/hybrid_batch.py`, measured in `docs/measurements/mtp/README.md` ("Issue #37"): the
decode sweep at 1 request runs at 1.14x of MTP off with the back-off after rejected drafts as designed above, and at
1.28x with it off (`LOCALS_MTP_BACKOFF=0`). The back-off was carried over from the loop that reads every pass, where a
draft pass cost the pipelining; here a pass without a draft costs what a pass with one costs, so it only loses the
draft's tokens. Decided by the owner on #37 (2026-11-10): the pipelined loop runs without the back-off; the loop that
reads every pass keeps it (`LOCALS_MTP_DRAFTS=2`, or `mtp_pipelined` off); `LOCALS_MTP_BACKOFF` sets both loops the same
way, which keeps the seeded regression test for either setting, each against the other loop at that setting. Two departures from the design above, made in the build and open to the
owner's veto: a forced span discards the whole pass launched after it (the design kept that pass's first slot and made
only the draft's a hole), one pass per span and simpler bookkeeping, with the agents bench unmoved; and prompt-lookup
drafts are not consulted inside a stretch (they need the CPU), only at the boundary before one. The seeded regression
test holds token for token on the tiny float32 model of `tests/test_mtp.py`; on the real float16 model the two loops
drift apart in most seeded runs (a 2-token pass rounds differently from the 1-token step the other loop runs in its
back-off, the effect `greedy_check.py` shows for greedy), so the real-model proof is the greedy one and the sampled
distribution, not the tokens. The estimate of 1.35x assumed a pass at 1.17x of a step; measured, a pass
costs the GPU about 11 ms against 7.6 for a step (the 2-token pass 9.4, the per-token states up to 0.6, the MTP layer
and its draft through the output head 1.6), and both loops are bound by the GPU, not by the CPU reads this record removed.
