# Vision and unload measurements

Run on an M2 Max (32 GB), 2026-10-31, at commit `20bafe1`, with `mlx-community/Qwen3.5-4B-MLX-4bit` and
`mlx-community/Qwen3-VL-8B-Instruct-4bit`. mlx-vlm 0.7.11 (mlx 0.32.4, transformers 5.22.1), run through `uvx`.
Each command is run from this folder. `<repo>` is the clone's folder; in the result files, `<repo>` and
`<scratch>` stand for the paths the run printed.

## Our answers against mlx-vlm's (`docs/server/README.md`, Images)

```
python make_pictures.py pics                                    # the six pictures and their questions
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>.json      # mlx-vlm, its own prompt
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>_nl.json nl  # with a newline after the image
<repo>/.venv/bin/python run_locals.py <model> pics loc_<m>.json [nodeepstack]  # ours, in process
<repo>/.venv/bin/python prompt_check.py <model> pics results/vlm_<m>.json      # our prompt against mlx-vlm's
python compare.py results                                       # ours against mlx-vlm's own prompt
```

Every question is asked twice; on every side both answers were the same. `run_locals.py nodeepstack` drops the
deepstack features the tower returns (it prints how many levels it saw: 3 for Qwen3-VL). Ours were run on
2026-10-31 at commit `929eb9f` with the image joined to the question without a newline (the change after it); the
mlx-vlm answers are the ones from commit `20bafe1`. `results/compare.txt` is the comparison and
`results/prompt_check.txt` the prompt check (the image's pad tokens counted as one).

Up to commit `929eb9f` our prompt had a newline after the image. Our answers from then are `loc_*_nl.json`, and
`results/compare_newline.txt` compares them with mlx-vlm given the same newline (`vlm_*_nl.json`).

## Unloading (`docs/server/README.md`, Unloading the model)

```
<repo>/.venv/bin/python unload_serve.py 8399 &                  # the server, its PID from $!
<repo>/.venv/bin/python unload_client.py 8399 <PID>
```

`unload_serve.py` starts `Server` and `serve` in Python, not `locals serve`, which saves the port and the model as
the next default. The output is `results/unload_run.txt`; `phys_footprint` is read with `footprint -p`.

## Prefill memory of a long prompt with one image (`docs/server/README.md`, Images)

```
python make_pictures.py pics
git archive 5b6a125 locals | tar -x -C before                   # the code before, beside the working tree's
PYTHONPATH=before <repo>/.venv/bin/python prefill_memory.py <model> pics/invoice_900x500.png mem before.json
<repo>/.venv/bin/python prefill_memory.py <model> pics/invoice_900x500.png mem after.json
```

`prefill_memory.py` prints which `locals` it imported. `mem` puts about 36,000 tokens of text before the picture;
`exact` asks the invoice question with `prefill_step` 256, so a chunk ends inside the image. Each process sends the
request twice, and both peaks were the same to the MB on every side. The output is `results/prefill_memory.txt`.

## `count_tokens` with images (`docs/server/README.md`, Images)

```
git archive c358179 locals | tar -x -C before                   # the code before the change
PYTHONPATH=before <repo>/.venv/bin/python unload_serve.py 8399 &  # from a folder outside the repo: the
<repo>/.venv/bin/python count_tokens_client.py 8399             # repo as the current folder wins over PYTHONPATH
PYTHONPATH=<repo> <repo>/.venv/bin/python unload_serve.py 8399 &  # after; each server stopped by its PID from $!
<repo>/.venv/bin/python count_tokens_client.py 8399
```

The client sends two 4032x3024 JPEGs (made in the script) to `count_tokens` seven times, then once to
`/v1/messages`, and prints both token counts and the median time. Each side was run twice, each by a fresh server.
The output is `results/count_tokens.txt`.

## Where an image request's time goes (`results/stage_cost.txt`, `results/mrope_cost.txt`)

```
<repo>/.venv/bin/python stage_cost.py <model>                   # vision.load, the tower, the prompt read, the request
<repo>/.venv/bin/python mrope_cost.py <model> mrope             # the prompt read with M-RoPE
<repo>/.venv/bin/python mrope_cost.py <model> stock             # with the layers' own rope: the floor for any M-RoPE speedup
```

Both send four requests of about 5,600 tokens (text, a 1280x800 and a 1024x1024 picture) and print the medians of
runs 2 to 4. That request is in `request.py`, which `stage_cost.py`, `mrope_cost.py`, `turn2_cost.py`,
`patch_cost.py` and `patch_round.py` import from this folder: a script run from a copy needs `request.py`
beside it. Each line of the results is one process; each model and mode was run twice.

## The patch embed and `positions` (`results/patch_cost.txt`, issue #26)

```
git archive fb669f8 locals | tar -x -C before                   # the code before: two frames a patch, a Conv3d over them
PYTHONPATH=before <repo>/.venv/bin/python patch_cost.py <model> before_<m>_<n>.npz   # from a folder outside the repo (see above)
<repo>/.venv/bin/python patch_cost.py <model> after_<m>_<n>.npz   # after: one frame a patch, the frame weights summed at load
<repo>/.venv/bin/python patch_compare.py before_<m>_1.npz before_<m>_2.npz   # the unchanged code twice: the noise floor
<repo>/.venv/bin/python patch_compare.py before_<m>_1.npz after_<m>_<n>.npz  # the tower's output, value for value
<repo>/.venv/bin/python patch_round.py <model>                  # the patch embed's own output against the conv in float32
<repo>/.venv/bin/python patch_form.py                           # the conv and the matmul alone, random weights
PYTHONPATH=before <repo>/.venv/bin/python run_locals.py <model> pics tok_before_<m>.json   # the six questions, before
<repo>/.venv/bin/python run_locals.py <model> pics tok_after_<m>.json                      # and after
<repo>/.venv/bin/python answers_compare.py tok_before_<m>.json tok_after_<m>.json   # exits 1 if any picture differs
```

`patch_cost.py` sends `stage_cost.py`'s request (both take it from `request.py`) and times, inside it,
`vision.preprocess` (the HTTP thread), the tower's patch embed alone (evaluated on its own, so its time is its own plus
the pixels' copy to the GPU), `vision.positions` (evaluated), the whole tower and the whole request; the prompt cache is emptied before each request, so every run runs
the tower. With a file name it saves the last run's tower features and positions, which `patch_compare.py` compares
between two builds; `answers_compare.py` compares two `run_locals.py` outputs answer for answer. Each build was run
twice per model, one model at a time, no other locals process running. `patch_cost.py`, `patch_round.py` and
`run_locals.py` print which `locals` they imported. The 8B model's files are in a cache on another drive
(`HF_HUB_CACHE`).

## A follow-up turn about the same images (`results/turn2_cost.txt`)

```
git archive 6b290df locals | tar -x -C before                   # the code before image prompts were cached
PYTHONPATH=before <repo>/.venv/bin/python turn2_cost.py <model>   # from a folder outside the repo (see above)
PYTHONPATH=<repo> <repo>/.venv/bin/python turn2_cost.py <model>
```

The same request as above is the first turn; the second adds its answer and a question. Each run empties the prompt
cache, asks the first turn, and times the second (one token). Each side was run twice per model.
