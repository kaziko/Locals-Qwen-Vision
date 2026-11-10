"""Encode time (mx.async_eval returns) and GPU time (mx.eval returns after it) of a bare 1-token step and a bare
2-token pass through the model body, on a one-row batch cache, with and without the recurrent per-token states.
Run from the repo root: .venv/bin/python docs/measurements/mtp/pass_cost.py <model>
"""
import sys, time, statistics as st, mlx.core as mx
from locals.runtime import Runtime
from locals import hybrid_batch as hb
from locals.kernels import head_logits
from mlx_lm.models.cache import ArraysCache
rt = Runtime.load(sys.argv[1])
lm = hb.text_model(rt.model)
eng = hb.HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4, share_prefix=False)
prompt = rt.tokenizer.encode("Explain how a refrigerator works, in about 150 words. " * 4)
cache = hb._merge([eng._prefill(prompt, [])[0]])
mx.eval([c.state for c in cache])
def measure(S, states, n=40):
    enc, gpu = [], []
    for i in range(n):
        for c in cache:
            if isinstance(c, ArraysCache):
                c.want_states, c.token_states = states, None
        t0 = time.perf_counter()
        h = lm.model(mx.array([[100 + i] * S]), cache)
        lg = head_logits(lm, h)
        y = mx.argmax(lg, axis=-1)
        t1 = time.perf_counter()
        mx.async_eval(y, [a for c in cache for a in (getattr(c, "keys", None), getattr(c, "values", None), *(getattr(c, "cache", None) or [])) if isinstance(a, mx.array)])
        t2 = time.perf_counter()
        mx.eval(y)
        t3 = time.perf_counter()
        for c in cache:
            if isinstance(c, ArraysCache):
                c.want_states = False
        enc.append((t1 - t0, t2 - t1)); gpu.append(t3 - t2)
    b, e = zip(*enc)
    return f"S={S} states={states}: build {1000*st.median(b):.2f} encode {1000*st.median(e):.2f} wait {1000*st.median(gpu):.2f} ms (sum {1000*(st.median(b)+st.median(e)+st.median(gpu)):.2f})"
measure(1, False, 5)
for S, states in ((1, False), (2, False), (2, True), (1, False), (2, True)):
    print(measure(S, states), flush=True)
