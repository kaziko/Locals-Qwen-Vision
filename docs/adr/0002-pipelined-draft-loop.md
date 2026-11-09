---
status: proposed
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

The reply is the same token for token: only the main model's own picks are committed, as now.

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

## Open points for the maintainer

1. Holes in the main KV, or a rewrite of the cache offset? This record proposes holes because the MTP entry
   already lives that way and the attention mask is one flag column.
2. Where to compact: only at the stretch's end (cheapest), or also whenever the CPU catches up with no backlog.
3. Sampled replies with a seed: the draws stay on the device, but `r.draws` counts per pass on the CPU; the
   count must advance in the same order as today for the same seed to give the same reply.
