"""Usage: patch_compare.py BEFORE.npz AFTER.npz — two patch_cost.py saves compared value for value: the max abs
difference of each picture's tower features (and the largest feature, for scale), and whether the positions are the
same array."""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in sorted(k for k in a.files if k.startswith("features")):
    x, y = a[k], b[k]  # each [k] reads the member from the zip again
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    print(f"{k} {x.shape}: max abs diff {d.max():.4g}, mean abs diff {d.mean():.4g}, "
          f"max abs value {np.abs(x).max():.4g}, {int((d > 0).sum())} of {d.size} values differ")
pa, pb = a["positions"], b["positions"]
print(f"positions {pa.shape}: {'the same array' if np.array_equal(pa, pb) else 'DIFFER'}")
