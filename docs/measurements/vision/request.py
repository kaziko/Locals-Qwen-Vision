"""The image request the timing scripts in this folder send: text, a 1280x800 and a 1024x1024 picture. The
pictures' noise is the first draws of one default_rng(0), so they are drawn once, in that order, and kept; the
results in results/ were measured on them. Imported from
this folder: Python puts the script's own folder first on sys.path."""
import base64
import functools
import io
import time

import mlx.core as mx
import numpy as np
from PIL import Image

_rng = np.random.default_rng(0)
FILLER = "".join(f"Line {i}: the parcel {i * 7919 % 10007} left the depot at {i % 24:02d}:{i * 13 % 60:02d}.\n" for i in range(150))


@functools.cache
def pictures():
    """The two PNGs, 1280x800 then 1024x1024. Drawn on the first call; later calls return the same bytes."""
    out = []
    for w, h in ((1280, 800), (1024, 1024)):
        y, x = np.mgrid[0:h, 0:w]
        a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1).astype(np.int16)
        a = np.clip(a + _rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
        b = io.BytesIO()
        Image.fromarray(a).save(b, "PNG")
        out.append(b.getvalue())
    return tuple(out)


def messages(question):
    """One user turn: the filler, the two pictures as data URLs, then ``question``."""
    pics = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p).decode()}}
            for p in pictures()]
    return [{"role": "user", "content": [{"type": "text", "text": FILLER}, *pics, {"type": "text", "text": question}]}]


def request(msgs=None, max_tokens=1):
    """The request the scripts time: by default the one-token request about both pictures."""
    if msgs is None:
        msgs = messages("Describe both pictures.")
    return {"model": "locals", "max_tokens": max_tokens, "temperature": 0, "messages": msgs}


def timer(stages, key, f, sync):
    """``f`` timed into ``stages[key]``; with ``sync``, the GPU is idle at the start and the result evaluated."""
    def w(*a, **k):
        if sync:
            mx.synchronize()
        t = time.perf_counter()
        r = f(*a, **k)
        if sync:
            mx.eval(r)
            mx.synchronize()
        stages[key].append(time.perf_counter() - t)
        return r
    return w


def empty_cache(srv):
    """Empty the prompt cache: a cached prompt is read no further, and its tower does not run."""
    if srv.service("locals")._lru is not None:
        srv.service("locals")._lru.trim_to(n_sequences=0)
