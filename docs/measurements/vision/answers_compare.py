"""Usage: answers_compare.py A.json B.json — two run_locals.py outputs compared: per picture, whether both answers are the same
in A and B, and the first 60 characters of each first answer. A picture B lacks, or has no answers for, is a DIFFER."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
names = [n for n in a if not n.startswith("_")]
same = 0
for name in names:
    x, y = a[name].get("answers") or [], (b.get(name) or {}).get("answers") or []
    if not y:
        verdict = "DIFFER (missing in B)"
    elif x == y:
        verdict = "same answers"
        same += 1
    else:
        verdict = "DIFFER"
    firsts = [f"{side} {v[0][:60]!r}" for side, v in (("before", x), ("after", y)) if v]
    print("; ".join([f"{name}: {verdict}"] + firsts))
print(f"{same} of {len(names)} pictures: the same answers before and after")
