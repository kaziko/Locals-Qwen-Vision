"""Usage: answers_compare.py A.json B.json — two run_locals.py outputs compared: per picture in A or B, whether both
answers are the same in A and B, and the first 60 characters of each first answer. A picture one side lacks, or has
no answers for, is a DIFFER that names the side. The exit status is 1 when any picture is not the same answers."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
names = [n for n in a if not n.startswith("_")] + [n for n in b if not n.startswith("_") and n not in a]
same = 0
for name in names:
    x, y = ((side.get(name) or {}).get("answers") or [] for side in (a, b))
    if not x or not y:
        verdict = f"DIFFER (missing in {'both' if not x and not y else 'A' if not x else 'B'})"
    elif x == y:
        verdict = "same answers"
        same += 1
    else:
        verdict = "DIFFER"
    firsts = [f"{side} {v[0][:60]!r}" for side, v in (("before", x), ("after", y)) if v]
    print("; ".join([f"{name}: {verdict}"] + firsts))
print(f"{same} of {len(names)} pictures: the same answers before and after")
sys.exit(0 if same == len(names) else 1)
