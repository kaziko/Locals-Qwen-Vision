"""Usage: patch_form.py — the patch embed by form, no model (issue #26): a two-frame Conv3d as the checkpoint has it, a
one-frame Conv3d, and one matmul over a (16, 16, 3) patch vector (nn.Linear), each on the same random weights and
pixels, with and without the pixels' copy to the GPU. 2024 patches (a 1280x800 and a 1024x1024 picture), hidden 1024."""
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

n, hid = 2024, 1024
dtype = mx.bfloat16
rng = np.random.default_rng(0)
one = rng.standard_normal((n, 16 * 16 * 3)).astype(np.float32)
two = np.concatenate([one, one], axis=1)  # the same frame twice, as the checkpoint's conv reads it

conv2 = nn.Conv3d(3, hid, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True)
conv1 = nn.Conv3d(3, hid, kernel_size=(1, 16, 16), stride=(1, 16, 16), bias=True)
lin = nn.Linear(768, hid, bias=True)
for m in (conv2, conv1, lin):
    m.set_dtype(dtype)
mx.eval(conv2.parameters(), conv1.parameters(), lin.parameters())


def timed(name, f, reps=20):
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        r = f()
        mx.eval(r)
        mx.synchronize()
        ts.append(time.perf_counter() - t)
    ts = sorted(ts)[2:-2]
    print(f"{name}: median {np.median(ts) * 1000:.2f} ms (min {min(ts) * 1000:.2f})")


timed("transfer two frames (np -> mx, cast)", lambda: mx.array(two).astype(dtype))
timed("transfer one frame (np -> mx, cast)", lambda: mx.array(one).astype(dtype))
x2 = mx.array(two).astype(dtype); x1 = mx.array(one).astype(dtype); mx.eval(x2, x1)


def conv_two(x):
    return conv2(x.reshape(-1, 2, 16, 16, 3)).reshape(n, -1)


timed("conv3d 2 frames (resident input)", lambda: conv_two(x2))
timed("conv3d 1 frame (resident input)", lambda: conv1(x1.reshape(-1, 1, 16, 16, 3)).reshape(n, -1))
timed("linear 768 (resident input)", lambda: lin(x1))
timed("transfer + conv3d 2 frames", lambda: conv_two(mx.array(two).astype(dtype)))
timed("transfer + linear 768", lambda: lin(mx.array(one).astype(dtype)))
