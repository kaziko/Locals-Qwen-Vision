"""docs/measurements/vision/answers_compare.py (issue #35): every picture in A or B gets a line, a picture one
side lacks, or has no answers for, is a DIFFER line that names the side, the summary counts both sides' pictures,
and the exit status is 1 when any picture is not the same answers."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "docs/measurements/vision/answers_compare.py"


def _run(tmp_path, a, b, code=0):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    r = subprocess.run([sys.executable, str(SCRIPT), str(pa), str(pb)], capture_output=True, text=True)
    assert r.returncode == code, r.stderr
    return r.stdout.splitlines()


A = {"cat.png": {"answers": ["cat 42", "cat 42"]}, "bars.png": {"answers": ["bars", "bars"]},
     "_deepstack": 3}


def test_the_same_answers_in_both_files(tmp_path):
    assert _run(tmp_path, A, A) == ["cat.png: same answers; before 'cat 42'; after 'cat 42'",
                                    "bars.png: same answers; before 'bars'; after 'bars'",
                                    "2 of 2 pictures: the same answers before and after"]


def test_a_picture_missing_in_b_is_a_differ_line_and_the_summary_prints(tmp_path):
    b = {"bars.png": A["bars.png"]}
    assert _run(tmp_path, A, b, code=1) == ["cat.png: DIFFER (missing in B); before 'cat 42'",
                                            "bars.png: same answers; before 'bars'; after 'bars'",
                                            "1 of 2 pictures: the same answers before and after"]


def test_a_picture_with_no_answers_prints_no_first_answer(tmp_path):
    b = {"cat.png": {"answers": []}, "bars.png": {"answers": ["other"]}}
    assert _run(tmp_path, A, b, code=1) == ["cat.png: DIFFER (missing in B); before 'cat 42'",
                                            "bars.png: DIFFER; before 'bars'; after 'other'",
                                            "0 of 2 pictures: the same answers before and after"]


def test_a_picture_only_in_b_is_a_missing_in_a_line_and_counts(tmp_path):
    b = dict(A, **{"receipt.png": {"answers": ["total 9"]}})
    assert _run(tmp_path, A, b, code=1) == ["cat.png: same answers; before 'cat 42'; after 'cat 42'",
                                            "bars.png: same answers; before 'bars'; after 'bars'",
                                            "receipt.png: DIFFER (missing in A); after 'total 9'",
                                            "2 of 3 pictures: the same answers before and after"]


def test_empty_in_a_is_missing_in_a_and_empty_on_both_sides_is_missing_in_both(tmp_path):
    a = {"cat.png": {"answers": []}, "bars.png": {"answers": []}}
    b = {"cat.png": {"answers": ["cat"]}, "bars.png": {}}
    assert _run(tmp_path, a, b, code=1) == ["cat.png: DIFFER (missing in A); after 'cat'",
                                            "bars.png: DIFFER (missing in both)",
                                            "0 of 2 pictures: the same answers before and after"]
