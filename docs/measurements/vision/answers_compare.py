"""Usage: answers_compare.py A.json B.json — two run_locals.py outputs compared: per picture, whether both answers are the same
in A and B, and the first 60 characters of each first answer."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
names = [n for n in a if not n.startswith("_")]
same = 0
for name in names:
    x, y = a[name]["answers"], b[name]["answers"]
    ok = x == y
    same += ok
    print(f"{name}: {'same answers' if ok else 'DIFFER'}; before {x[0][:60]!r}; after {y[0][:60]!r}")
print(f"{same} of {len(names)} pictures: the same answers before and after")
