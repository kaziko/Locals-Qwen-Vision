"""The measurement scripts' shared setup (docs/measurements/timing_setup.py, issue #25): while a script's busy
marker runs, another Locals process sees the GPU as busy, and no longer once it stops; inside a timed section the
marker does nothing."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from locals import kernels  # noqa: E402

PATH = Path(__file__).parent.parent / "docs/measurements/timing_setup.py"


def _module():
    spec = importlib.util.spec_from_file_location("timing_setup", PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CHILD = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("timing_setup", {str(PATH)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
b = m.BusyMarker(); b.start(); print("on", flush=True)
sys.stdin.readline(); b.stop(); print("off", flush=True)
sys.stdin.readline()
"""


def _wait(cond, limit):
    end = time.time() + limit
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def test_another_process_sees_the_gpu_busy_while_the_marker_runs():
    p = subprocess.Popen([sys.executable, "-c", CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "on"
        assert _wait(kernels.others_busy, 2.0)  # (its first mark is at start)
        time.sleep(4.0)  # longer than the 3 s window: the marks go on
        assert kernels.others_busy()
        p.stdin.write("\n")
        p.stdin.flush()
        assert p.stdout.readline().strip() == "off"
        assert _wait(lambda: not kernels.others_busy(), 3.5)  # the process is alive: only the window ends it
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()


def test_the_marker_marks_nothing_inside_a_timed_section():
    b = _module().BusyMarker()
    path = kernels._busy_file(os.getpid())
    b.start()
    try:
        with b.quiet():
            before = os.stat(path).st_mtime  # (quiet marks once on the way in)
            time.sleep(2.5)
            assert os.stat(path).st_mtime == before
        assert _wait(lambda: os.stat(path).st_mtime > before, 1.5)  # marks again after it
    finally:
        b.stop()
