"""docs/measurements/vision/answers_compare.py (issue #35): a picture that B lacks, or that has no answers, is a
DIFFER line, and every picture line and the summary still print."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "docs/measurements/vision/answers_compare.py"


def _run(tmp_path, a, b):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    r = subprocess.run([sys.executable, str(SCRIPT), str(pa), str(pb)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


A = {"cat.png": {"answers": ["cat 42", "cat 42"]}, "bars.png": {"answers": ["bars", "bars"]},
     "_deepstack": 3}


def test_the_same_answers_in_both_files(tmp_path):
    assert _run(tmp_path, A, A) == ["cat.png: same answers; before 'cat 42'; after 'cat 42'",
                     "bars.png: same answers; before 'bars'; after 'bars'",
                     "2 of 2 pictures: the same answers before and after"]


def test_a_picture_missing_in_b_is_a_differ_line_and_the_summary_prints(tmp_path):
    b = {"bars.png": A["bars.png"]}
    assert _run(tmp_path, A, b) == ["cat.png: DIFFER (missing in B); before 'cat 42'",
                                    "bars.png: same answers; before 'bars'; after 'bars'",
                                    "1 of 2 pictures: the same answers before and after"]


def test_a_picture_with_no_answers_prints_no_first_answer(tmp_path):
    b = {"cat.png": {"answers": []}, "bars.png": {"answers": ["other"]}}
    assert _run(tmp_path, A, b) == ["cat.png: DIFFER (missing in B); before 'cat 42'",
                                    "bars.png: DIFFER; before 'bars'; after 'other'",
                                    "0 of 2 pictures: the same answers before and after"]
