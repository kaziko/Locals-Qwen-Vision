"""A GPU without tuned kernels: shapes run on MLX's kernel at once, and are tuned one at a
time while the server is idle (no model needed: a small quantized layer)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from locals import kernels


def test_untuned_shapes_wait_for_the_idle_tuner(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    model = nn.Sequential(layer)
    work = kernels.pending_work(model, max_m=3)
    assert [m for _, m in work] == [1, 2, 3, kernels.PREFILL_TUNE_M]  # smallest batches first, then how prompts are read
    x = mx.zeros((1, 512), layer["scales"].dtype)
    assert kernels._choose(x, layer) == ("mlx",)  # untuned: MLX's kernel, nothing tuned in the request
    key = "1024x512x1" + ("" if layer["scales"].dtype == mx.float16 else "x" + str(layer["scales"].dtype).split(".")[-1])
    assert key not in kernels._load_tune()
    picked = kernels.tune_one(*work[0])
    assert kernels._load_tune()[key] == picked and kernels._choose(x, layer) == picked
    assert [m for _, m in kernels.pending_work(model, max_m=3)] == [2, 3, kernels.PREFILL_TUNE_M]
    assert (tmp_path / "tune.json").exists()


def _fp16_layer():
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    layer.set_dtype(mx.float16)
    return layer


def test_how_prompts_are_read_is_timed_on_this_gpu(tmp_path, monkeypatch):
    """Dequantize + dense GEMM or MLX's quantized GEMM for a prompt: timed once per shape (MLX's uses the M5's
    neural accelerators; on an M2 Pro the dense one is 3-11% faster), untimed in a request on the server."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_DENSE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    monkeypatch.delenv("LOCALS_DENSE_PREFILL", raising=False)
    layer = _fp16_layer()
    x = mx.zeros((256, 512), mx.float16)
    assert kernels._dense_ok(x, layer)  # not timed yet: dense, as before
    key = kernels._prefill_key(1024, 512, "float16")
    assert key not in kernels._load_tune()
    kernels._DENSE.clear()
    picked = kernels.tune_one(layer, kernels.PREFILL_TUNE_M)
    tune = kernels._load_tune()
    assert tune[key] == picked and set(tune["__times__"][key]) == {"mlx", "dense"}
    assert kernels._dense_ok(x, layer) == (picked == ("dense",))
    assert not kernels._dense_ok(mx.zeros((64, 512), mx.float16), layer)  # few rows: never dense
    assert kernels.PREFILL_TUNE_M not in [m for _, m in kernels.pending_work(nn.Sequential(layer), max_m=0)]
    xr = mx.random.normal((256, 512), key=mx.random.key(3)).astype(mx.float16)
    ref = mx.quantized_matmul(xr, layer["weight"], layer["scales"], layer["biases"], transpose=True, group_size=64, bits=4)
    for dense in (True, False):  # either way, the layer computes the same
        kernels._DENSE[(1024, 512, "float16")] = dense
        y = kernels._linear_call_inner(layer, xr)
        assert float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)))) < 0.05 * (1 + float(mx.max(mx.abs(ref))))


def test_the_prompt_reading_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    layer, x = _fp16_layer(), mx.zeros((256, 512), mx.float16)
    for value, dense in (("0", False), ("1", True)):
        monkeypatch.setattr(kernels, "_DENSE", {})
        monkeypatch.setenv("LOCALS_DENSE_PREFILL", value)
        assert kernels._dense_ok(x, layer) is dense
    assert not (tmp_path / "tune.json").exists()  # nothing timed


def test_the_tuning_report(tmp_path, monkeypatch):
    """locals tune --report: per batch size MLX's kernels against the chosen ones, and how prompts are read."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DENSE", {})
    monkeypatch.setattr(kernels, "_DEFER", False)
    monkeypatch.delenv("LOCALS_DENSE_PREFILL", raising=False)
    layer = _fp16_layer()
    model = nn.Sequential(layer, _fp16_layer())  # one shape, two layers
    for mod, M in kernels.pending_work(model, max_m=4):
        kernels.tune_one(mod, M)
    text = kernels.tune_report(model)
    assert "2 quantized layers in 1 shapes" in text
    rows = {line.split()[0]: line for line in text.splitlines() if line[:4].strip().isdigit()}
    assert "x" in rows["4"] and "ms" in rows["4"] and "not timed" in rows["8"]
    assert "1024x512 (2 layers): MLX" in text


def test_an_idle_tuner_waits_while_another_server_works(tmp_path, monkeypatch):
    """A server tuning while idle saw only its own engines: next to another locals server serving on the same GPU it
    timed kernels anyway, and kept the skewed timings. Each busy process marks a file; the tuner waits for them."""
    import os
    import subprocess
    import time

    from locals import service

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_BUSY_DIR", str(tmp_path))
    other = tmp_path / f"busy-{os.getppid()}"  # a live process, busy a moment ago
    other.write_text("")
    assert kernels.others_busy()
    os.utime(other, (time.time() - 10, time.time() - 10))
    assert not kernels.others_busy()  # its last mark is old: idle now
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / f"busy-{dead.pid}").write_text("")
    assert not kernels.others_busy() and not (tmp_path / f"busy-{dead.pid}").exists()
    kernels.mark_busy(now=True)
    assert (tmp_path / f"busy-{os.getpid()}").exists() and not kernels.others_busy()  # (its own: not another's)

    svc = service.BatchService.__new__(service.BatchService)
    svc._tune_work, svc._inflight = [("layer", 1)], {}
    tuned = []
    monkeypatch.setattr(kernels, "tune_one", lambda mod, M: tuned.append((mod, M)))
    os.utime(other, None)
    svc._tune_while_idle()
    assert tuned == [] and svc._tune_work == [("layer", 1)]  # left for later, not dropped
    os.utime(other, (time.time() - 10, time.time() - 10))
    svc._tune_after = 0.0
    svc._tune_while_idle()
    assert tuned == [("layer", 1)] and not svc._tune_work


def test_shapes_timed_under_another_mlx_are_timed_again(tmp_path, monkeypatch):
    """The tune cache was not keyed on MLX's version: choices timed under an older MLX stayed for good after an
    upgrade. They are stale now: still used, and timed again by the idle tuner (and locals tune)."""
    import json

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    dt = str(layer["scales"].dtype).split(".")[-1]
    key = "1024x512x1" + ("" if dt == "float16" else f"x{dt}")
    (tmp_path / "tune.json").write_text(json.dumps({"__mlx__": "0.0.1", key: ["mlx"], "__times__": {key: {"mlx": 1.0}}}))
    tune = kernels._load_tune()
    assert kernels._stale_keys(tune) == [key]  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    assert json.loads((tmp_path / "tune.json").read_text())["__timed__"] == {key: "0.0.1"}
    x = mx.zeros((1, 512), layer["scales"].dtype)
    assert kernels._choose(x, layer) == ("mlx",)  # still used, not timed in a request
    work = kernels.pending_work(nn.Sequential(layer), max_m=1)
    assert [m for _, m in work] == [1, kernels.PREFILL_TUNE_M]
    kernels.tune_one(*work[0])
    assert key not in kernels._stale_keys(kernels._load_tune()) and kernels._load_tune()["__times__"][key] != {"mlx": 1.0}
    assert json.loads((tmp_path / "tune.json").read_text())["__timed__"][key] == mx.__version__  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    assert [m for _, m in kernels.pending_work(nn.Sequential(layer), max_m=1)] == [kernels.PREFILL_TUNE_M]


def test_a_cache_from_before_versions_gets_one_and_nothing_is_stale(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    (tmp_path / "tune.json").write_text(json.dumps({"1024x512x1": ["mlx"]}))
    tune = kernels._load_tune()
    assert not kernels._stale_keys(tune)
    saved = json.loads((tmp_path / "tune.json").read_text())
    assert saved["__timed__"] == {"1024x512x1": mx.__version__} and "__mlx__" not in saved  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    # (the __stale__ intersection of a merge is gone with the list: test_a_merge_takes_each_shape_from_the_side_...)


def _old_version_file(tmp_path, monkeypatch, stale):
    """A one-file cache (``LOCALS_TUNE_CACHE``) last written under another MLX, with three timed shapes."""
    import json

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    data = {"__mlx__": "0.0.1", "1024x512x1": ["mlx"], "1024x512x2": ["mlx"], "1024x512xP": ["dense"]}
    if stale is not None:
        data["__stale__"] = stale
    path.write_text(json.dumps(data))
    return path


def test_a_version_change_marks_every_timed_shape_stale_whatever_the_file_listed(tmp_path, monkeypatch, capsys):
    """The save in the same load intersected the new marks with the file's ``__stale__`` list: a file that held one
    (even ``[]``) kept that list, so a real MLX upgrade marked nothing new, and the log line counted shapes that
    were not saved as stale."""
    import json

    for stale in (None, ["1024x512x2"], []):
        path = _old_version_file(tmp_path, monkeypatch, stale)
        tune = kernels._load_tune()
        every = ["1024x512x1", "1024x512x2", "1024x512xP"]
        assert kernels._stale_keys(tune) == every, stale  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
        saved = json.loads(path.read_text())
        assert kernels._stale_keys(saved) == every and "__stale__" not in saved and "__mlx__" not in saved, stale
        assert "the 3 kernel shapes" in capsys.readouterr().err, stale  # (the count that was saved)


def test_a_save_takes_a_shape_the_other_process_timed_again_under_this_mlx(tmp_path, monkeypatch):
    """The other process timed 1024x512x2 again and saved; this one still holds it as stale when it saves. (Issue #31:
    this test checked that a merge kept 1024x512x2 off the ``__stale__`` list; the list is gone, and the shape, its
    choice and its version come from the side that timed it under this MLX.)"""
    import json

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_PRESET", {})
    shapes = {"1024x512x1": ["mlx"], "1024x512x2": ["frag", 2, 2, 1]}
    path.write_text(json.dumps({**shapes, "__timed__": {"1024x512x1": "0.0.1", "1024x512x2": mx.__version__}}))
    monkeypatch.setattr(kernels, "_TUNE", {"1024x512x1": ("mlx",), "1024x512x2": ("mlx",),
                                           "__timed__": {"1024x512x1": "0.0.1", "1024x512x2": "0.0.1"}})
    kernels._save_tune()
    saved = json.loads(path.read_text())
    assert saved["1024x512x2"] == ["frag", 2, 2, 1] and kernels._stale_keys(saved) == ["1024x512x1"]
    assert kernels._TUNE["1024x512x2"] == ("frag", 2, 2, 1) and kernels._stale_keys(kernels._TUNE) == ["1024x512x1"]


def _version_dir(tmp_path, monkeypatch):
    """The default cache directory, as without ``LOCALS_TUNE_CACHE``: this MLX's file is absent."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / f"{kernels._TUNE_STEM}_mlx{mx.__version__}.json"))
    monkeypatch.setattr(kernels, "_TUNE_PATH_FROM_ENV", None)
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    return kernels._TUNE_STEM


def test_a_new_mlx_version_seeds_its_file_from_the_old_shared_one(tmp_path, monkeypatch, capsys):
    """Issue #31: a seed was stale whole, even where it said this MLX timed it. Now it converts as any old file does:
    the shape its ``__stale__`` list names is stale, the one its ``__mlx__`` (this MLX) covers is not."""
    import json

    _version_dir(tmp_path, monkeypatch)
    old = tmp_path / kernels._TUNE_FILE
    old.write_text(json.dumps({"__mlx__": mx.__version__, "1024x512x1": ["frag", 2, 2, 1], "1024x512xP": ["dense"],
                               "__stale__": ["1024x512xP"], "__times__": {"1024x512x1": {"frag": 3.0}}}))
    before = old.read_bytes()
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 2, 2, 1) and tune["1024x512xP"] == ("dense",)  # the choices stay
    assert kernels._stale_keys(tune) == ["1024x512xP"]
    saved = json.loads(open(kernels._TUNE_PATH).read())
    assert saved["__timed__"] == {"1024x512x1": mx.__version__, "1024x512xP": kernels._STALE_MARK}
    assert saved["1024x512x1"] == ["frag", 2, 2, 1] and "__stale__" not in saved and "__mlx__" not in saved
    assert old.read_bytes() == before
    assert "the 1 kernel shapes" in capsys.readouterr().err


def test_the_seed_is_the_other_version_file_written_last(tmp_path, monkeypatch):
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    for i, (version, choice) in enumerate([("0.0.2", ["frag", 4, 2, 1]), ("0.0.1", ["mlx"]), (None, ["dense"])]):
        name = f"{stem}_mlx{version}.json" if version else f"{stem}.json"
        (tmp_path / name).write_text(json.dumps({"__mlx__": version or "0.0.0", "1024x512x1": choice}))
        os.utime(tmp_path / name, (1_000_000 + 100 * (3 - i),) * 2)  # 0.0.2's written last, the shared one first
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 4, 2, 1) and kernels._stale_keys(tune) == ["1024x512x1"]  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)


def test_a_seed_that_does_not_parse_is_passed_over_for_the_next_newest(tmp_path, monkeypatch):
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    good, cut = tmp_path / f"{stem}.json", tmp_path / f"{stem}_mlx0.0.2.json"
    good.write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    cut.write_text('{"1024x512x1": ["fr')  # (a write cut short)
    os.utime(good, (1_000_000,) * 2)
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("dense",) and kernels._stale_keys(tune) == ["1024x512x1"]  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    assert cut.read_text() == '{"1024x512x1": ["fr'  # (another version's file: not set aside from here)


def test_a_seed_file_that_goes_during_the_search_is_passed_over(tmp_path, monkeypatch):
    """Another process sets its bad file aside (renames it to .bad) after the search saw it: the load went on, not
    raised FileNotFoundError."""
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    good, gone = tmp_path / f"{stem}.json", tmp_path / f"{stem}_mlx0.0.2.json"
    good.write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    gone.write_text(json.dumps({"__mlx__": "0.0.2", "1024x512x1": ["mlx"]}))
    getmtime = os.path.getmtime

    def getmtime_then_renamed(p):
        t = getmtime(p)
        if p == str(gone):
            os.replace(gone, str(gone) + ".bad")  # (the other process, right after this one looked)
        return t

    monkeypatch.setattr(os.path, "getmtime", getmtime_then_renamed)
    assert kernels._seed()[0] == str(good)
    monkeypatch.setattr(kernels, "_TUNE", None)
    assert kernels._load_tune()["1024x512x1"] == ("dense",)


def test_a_bad_file_for_this_mlx_is_set_aside_and_the_load_seeds_as_a_first_start(tmp_path, monkeypatch, capsys):
    """The bad file was renamed and a preset-only file saved in its place: from then on the file was there, so no
    seed ran, and every shape was timed from scratch."""
    import json

    stem = _version_dir(tmp_path, monkeypatch)
    (tmp_path / f"{stem}_mlx0.0.1.json").write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    mine = tmp_path / f"{stem}_mlx{mx.__version__}.json"
    mine.write_text('{"1024x512x1": ["fr')  # (a write cut short)
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("dense",) and kernels._stale_keys(tune) == ["1024x512x1"]  # (timed under 0.0.1)
    assert (tmp_path / f"{mine.name}.bad").read_text() == '{"1024x512x1": ["fr'
    assert json.loads(mine.read_text())["1024x512x1"] == ["dense"]
    assert "the 1 kernel shapes" in capsys.readouterr().err


def test_a_crash_marker_under_the_name_from_before_versioned_files_counts_once(tmp_path, monkeypatch):
    """A marker a dead process left under the shared name (before the file was per MLX version) was never read, and
    it stayed in ~/.cache/locals for good, with that name's lock file."""
    import fcntl
    import subprocess

    _version_dir(tmp_path, monkeypatch)
    dead = subprocess.Popen(["true"])
    dead.wait()
    old = tmp_path / kernels._TUNE_FILE
    (tmp_path / f"{old.name}.tuning-{dead.pid}").write_text("2560x4096x8")
    lock = tmp_path / f"{old.name}.lock"
    lock.write_text("")
    with open(lock) as held:  # (an older locals saving right now: its lock stays)
        fcntl.flock(held, fcntl.LOCK_EX)
        assert kernels._load_tune()["__crashed__"] == {"2560x4096x8": 1}
        assert not list(tmp_path.glob("*.tuning-*")) and lock.exists()
    monkeypatch.setattr(kernels, "_TUNE", None)
    assert kernels._load_tune()["__crashed__"] == {"2560x4096x8": 1}  # (once: the marker went)
    assert not lock.exists()


def test_a_bad_file_that_cannot_be_set_aside_is_not_seeded_over(tmp_path, monkeypatch):
    """The seed's save read the file it could not rename as empty, and replaced it: no .bad copy was left."""
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    (tmp_path / f"{stem}_mlx0.0.1.json").write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    mine = tmp_path / f"{stem}_mlx{mx.__version__}.json"
    mine.write_text('{"1024x512x1": ["fr')
    replace = os.replace

    def no_bad(a, b):
        if str(b).endswith(".bad"):
            raise PermissionError(b)
        return replace(a, b)

    monkeypatch.setattr(os, "replace", no_bad)
    assert "1024x512x1" not in kernels._load_tune()
    assert mine.read_text() == '{"1024x512x1": ["fr'


def test_a_bad_file_another_process_set_aside_first_is_not_stuck(tmp_path, monkeypatch):
    """The rename failed because another process had already moved the file to .bad: nothing was stuck, but the
    load skipped its seed and its save."""
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    (tmp_path / f"{stem}_mlx0.0.1.json").write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    mine = tmp_path / f"{stem}_mlx{mx.__version__}.json"
    mine.write_text('{"1024x512x1": ["fr')
    replace = os.replace

    def other_first(a, b):
        if str(b).endswith(".bad"):
            replace(a, b)  # (the other process)
            raise FileNotFoundError(a)
        return replace(a, b)

    monkeypatch.setattr(os, "replace", other_first)
    assert kernels._load_tune()["1024x512x1"] == ("dense",)
    assert json.loads(mine.read_text())["1024x512x1"] == ["dense"]


def _stuck_bad_file(tmp_path, monkeypatch):
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    mine = tmp_path / f"{stem}_mlx{mx.__version__}.json"
    mine.write_text('{"1024x512x1": ["fr')
    replace = os.replace

    def no_bad(a, b):
        if str(b).endswith(".bad"):
            raise PermissionError(b)
        return replace(a, b)

    monkeypatch.setattr(os, "replace", no_bad)
    return mine


def test_a_bad_file_that_cannot_be_set_aside_is_not_written_over_by_a_later_save(tmp_path, monkeypatch):
    """The guard was on the load's save only: the first shape the idle tuner timed saved over the bad file."""
    import pytest

    mine = _stuck_bad_file(tmp_path, monkeypatch)
    kernels._load_tune()["2560x4096x8"] = ("mlx",)
    with pytest.raises(OSError):
        kernels._save_tune()
    assert mine.read_text() == '{"1024x512x1": ["fr'


def test_crash_markers_stay_while_no_save_can_keep_their_count(tmp_path, monkeypatch):
    """The load deleted the markers but could not save the count: one crash per start, never the two that pin."""
    import subprocess

    mine = _stuck_bad_file(tmp_path, monkeypatch)
    for _ in range(2):
        dead = subprocess.Popen(["true"])
        dead.wait()
        (tmp_path / f"{mine.name}.tuning-{dead.pid}").write_text("2560x4096x8")
        monkeypatch.setattr(kernels, "_TUNE", None)
        tune = kernels._load_tune()
    assert tune["__crashed__"] == {"2560x4096x8": 2} and tune["2560x4096x8"] == ("mlx",)
    assert len(list(tmp_path.glob("*.tuning-*"))) == 2


def test_a_crash_under_the_old_name_and_one_under_this_versions_pin_the_shape(tmp_path, monkeypatch):
    import os
    import subprocess

    _version_dir(tmp_path, monkeypatch)
    dead = [subprocess.Popen(["true"]) for _ in range(2)]
    for p in dead:
        p.wait()
    (tmp_path / f"{kernels._TUNE_FILE}.tuning-{dead[0].pid}").write_text("2560x4096x8")
    (tmp_path / f"{os.path.basename(kernels._TUNE_PATH)}.tuning-{dead[1].pid}").write_text("2560x4096x8")
    tune = kernels._load_tune()
    assert tune["__crashed__"] == {"2560x4096x8": 2} and tune["2560x4096x8"] == ("mlx",)


def test_a_shape_pinned_after_two_crashes_is_never_marked_stale(tmp_path, monkeypatch):
    """A shape that stopped locals twice keeps MLX's kernel (_crashed_shapes). Marked stale, the idle tuner timed it
    again under the MLX that had aborted twice, and nothing took it off the list, so the server aborted at every
    start. Neither a version change, nor a seed, nor its second crash leaves it stale."""
    import json
    import subprocess

    pinned, timed = "1024x512x1", "1024x512x2"
    for seeded in (False, True):
        if seeded:
            _version_dir(tmp_path, monkeypatch)
            path = tmp_path / kernels._TUNE_FILE
        else:
            path = _old_version_file(tmp_path, monkeypatch, None)
        path.write_text(json.dumps({"__mlx__": "0.0.1", pinned: ["mlx"], timed: ["mlx"], "__crashed__": {pinned: 2}}))
        assert kernels._stale_keys(kernels._load_tune()) == [timed], seeded  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    path = tmp_path / "tune.json"  # this MLX's own file lists a shape stale, and its second crash comes now
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE_PATH_FROM_ENV", str(path))
    monkeypatch.setattr(kernels, "_TUNE", None)
    path.write_text(json.dumps({"__mlx__": mx.__version__, pinned: ["frag", 2, 2, 1], timed: ["mlx"],
                                "__stale__": [pinned, timed], "__crashed__": {pinned: 1}}))
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / f"tune.json.tuning-{dead.pid}").write_text(pinned)
    tune = kernels._load_tune()
    assert tune["__crashed__"] == {pinned: 2} and kernels._stale_keys(tune) == [timed]  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    assert kernels._stale_keys(json.loads(path.read_text())) == [timed]


def test_a_seed_this_mlx_wrote_that_never_saw_a_version_change_keeps_its_shapes(tmp_path, monkeypatch, capsys):
    """One install, one MLX: the old shared file names this MLX and has no ``__stale__`` list (only a version change
    writes one), so every shape in it was timed under this MLX. The first start under versioned names re-timed them
    all."""
    import json

    _version_dir(tmp_path, monkeypatch)
    (tmp_path / kernels._TUNE_FILE).write_text(json.dumps({"__mlx__": mx.__version__, "1024x512x1": ["frag", 2, 2, 1]}))
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 2, 2, 1) and not kernels._stale_keys(tune)  # (issue #31: each shape records its MLX version in __timed__; no __stale__ list)
    saved = json.loads(open(kernels._TUNE_PATH).read())  # (this MLX's file is written: the next start reads it)
    assert saved["1024x512x1"] == ["frag", 2, 2, 1] and saved["__timed__"] == {"1024x512x1": mx.__version__}
    assert "kernel shapes" not in capsys.readouterr().err


def _old_file(tmp_path, monkeypatch, data):
    """A one-file cache (``LOCALS_TUNE_CACHE``) in the format from before per-shape versions."""
    import json

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE_PATH_FROM_ENV", str(path))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    path.write_text(json.dumps(data))
    return path


def test_an_old_file_converts_to_per_shape_versions(tmp_path, monkeypatch):
    """Issue #31: a shape listed in ``__stale__`` stays stale; the others take the file's ``__mlx__`` (stale when it
    is another version); a file with no ``__mlx__`` has nothing stale. The choices stay, and ``__stale__`` and
    ``__mlx__`` are not written again."""
    import json

    shapes = {"1024x512x1": ["mlx"], "1024x512x2": ["frag", 2, 2, 1], "1024x512xP": ["dense"]}
    for was, stale, expect in [("0.0.1", ["1024x512x1"], ["1024x512x1", "1024x512x2", "1024x512xP"]),
                               (mx.__version__, ["1024x512x1"], ["1024x512x1"]),
                               (None, [], [])]:
        data = {**shapes, "__stale__": stale, **({"__mlx__": was} if was else {})}
        path = _old_file(tmp_path, monkeypatch, data)
        tune = kernels._load_tune()
        assert kernels._stale_keys(tune) == expect, was
        assert {k: list(tune[k]) for k in shapes} == shapes, was  # (the choices stay)
        saved = json.loads(path.read_text())
        assert "__stale__" not in saved and "__mlx__" not in saved, was
        monkeypatch.setattr(kernels, "_TUNE", None)
        assert kernels._stale_keys(kernels._load_tune()) == expect, was  # (the saved file says the same)


def test_a_merge_takes_each_shape_from_the_side_that_timed_it_under_this_mlx():
    """Issue #31: ``theirs`` timed A under this MLX, ``mine`` holds A from another version: A's choice, timings and
    version are ``theirs``'. Both or neither under this MLX: ``mine``'s, with ``mine``'s timings (none, if it has
    none), never half of each side."""
    now, old = mx.__version__, "0.0.1"
    mine = {"A": ("mlx",), "B": ("mlx",), "C": ("mlx",), "__times__": {"A": {"mlx": 9.0}, "C": {"mlx": 5.0}},
            "__timed__": {"A": old, "B": now, "C": old}}
    theirs = {"A": ("dense",), "B": ("dense",), "C": ("dense",), "D": ("dense",),
              "__times__": {"A": {"dense": 1.0}, "B": {"dense": 2.0}, "C": {"dense": 3.0}},
              "__timed__": {"A": now, "B": now, "C": old, "D": old}}
    shape = {k: f"1024x512x{i + 1}" for i, k in enumerate("ABCD")}  # (the keys a tuning file holds for shapes)

    def keyed(d):
        return {shape.get(k, k): ({shape.get(kk, kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in d.items()}

    out = kernels._merge_tune(keyed(mine), keyed(theirs))
    A, B, C, D = (shape[k] for k in "ABCD")
    assert (out[A], out["__times__"][A], out["__timed__"][A]) == (("dense",), {"dense": 1.0}, now)
    assert (out[B], B in out["__times__"], out["__timed__"][B]) == (("mlx",), False, now)  # (both: mine)
    assert (out[C], out["__times__"][C], out["__timed__"][C]) == (("mlx",), {"mlx": 5.0}, old)  # (neither: mine)
    assert (out[D], out["__timed__"][D]) == (("dense",), old)  # (only theirs has it)
    assert "__stale__" not in kernels._merge_tune({"__stale__": ["a"]}, {"__stale__": ["a", "b"]})


def test_two_mlx_versions_on_one_file_keep_each_others_fresh_shapes(tmp_path, monkeypatch):
    """Issue #31: two processes of two MLX versions save one ``LOCALS_TUNE_CACHE`` file in turn. Each save wrote its
    own ``__mlx__`` and ``__stale__`` list, so each marked the other's freshly timed shapes stale."""
    import json

    path = _old_file(tmp_path, monkeypatch, {})
    monkeypatch.setattr(kernels, "_PRESET", {})
    A, B, C = "1024x512x1", "1024x512x2", "1024x512x3"

    def process(version, tune=None):
        monkeypatch.setattr(kernels.mx, "__version__", version)
        monkeypatch.setattr(kernels, "_TUNE", tune)
        return kernels._load_tune()

    def time(tune, key):  # (what timing a shape writes: its choice and the MLX it was timed under)
        tune[key] = ("mlx",)
        tune.setdefault("__timed__", {})[key] = kernels.mx.__version__
        kernels._save_tune()

    one = process("0.0.1")
    time(one, A)
    two = process("0.0.2")
    time(two, B)
    time(process("0.0.1", one), C)
    time(process("0.0.2", two), B)
    saved = json.loads(path.read_text())
    assert saved["__timed__"] == {A: "0.0.1", B: "0.0.2", C: "0.0.1"}
    assert "__stale__" not in saved and "__mlx__" not in saved
    assert kernels._stale_keys(process("0.0.1")) == [B]  # (only what the other one timed)
    assert kernels._stale_keys(process("0.0.2")) == [A, C]


def test_a_file_with_per_shape_versions_ignores_a_stale_list_and_a_pin_is_never_stale(tmp_path, monkeypatch):
    """Issue #31: an older locals on a shared file can write a ``__stale__`` list next to ``__timed__``: it counts for
    nothing, and the next save leaves it out. A shape pinned after two crashes is not stale, whatever its version."""
    import json

    fresh, old, pinned = "1024x512x1", "1024x512x2", "1024x512x3"
    path = _old_file(tmp_path, monkeypatch, {
        fresh: ["mlx"], old: ["mlx"], pinned: ["mlx"], "__stale__": [fresh], "__mlx__": "0.0.1",
        "__timed__": {fresh: mx.__version__, old: "0.0.1", pinned: "0.0.1"}, "__crashed__": {pinned: 2}})
    assert kernels._stale_keys(kernels._load_tune()) == [old]
    saved = json.loads(path.read_text())
    assert "__stale__" not in saved and "__mlx__" not in saved and kernels._stale_keys(saved) == [old]


def test_a_merge_takes_a_shapes_family_winners_from_the_same_side():
    """A shape taken from ``theirs`` kept ``mine``'s per-family winners (``key|family``, which whole-step tuning
    reads): the choice from one side, the winners from the other."""
    A = "1024x512x1"
    mine = {A: ("mlx",), f"{A}|frag": ("frag", 2, 2, 1), f"{A}|mlx": ("mlx",), "__timed__": {A: "0.0.1"}}
    theirs = {A: ("frag", 4, 2, 1), f"{A}|frag": ("frag", 4, 2, 1), "__timed__": {A: mx.__version__}}
    out = kernels._merge_tune(mine, theirs)
    assert out[A] == ("frag", 4, 2, 1) and out[f"{A}|frag"] == ("frag", 4, 2, 1) and f"{A}|mlx" not in out


def test_one_file_with_per_shape_versions_logs_its_stale_shapes_after_an_mlx_upgrade(tmp_path, monkeypatch, capsys):
    """One ``LOCALS_TUNE_CACHE`` file, first opened under a new MLX: the log line said so through the file's
    ``__mlx__``, which is gone. It comes while no shape in the file was timed under this MLX."""
    import json

    A, B = "1024x512x1", "1024x512x2"
    path = _old_file(tmp_path, monkeypatch, {A: ["mlx"], B: ["mlx"], "__timed__": {A: "0.0.1", B: "0.0.1"}})
    kernels._load_tune()
    assert "MLX is 0" in (err := capsys.readouterr().err) and "(0.0.1 when" in err and "the 2 kernel shapes" in err
    data = json.loads(path.read_text())
    data["__timed__"][A] = mx.__version__  # (one shape timed again under this MLX)
    path.write_text(json.dumps(data))
    monkeypatch.setattr(kernels, "_TUNE", None)
    assert kernels._stale_keys(kernels._load_tune()) == [B]
    assert "kernel shapes" not in capsys.readouterr().err  # (said once, not at every start)


def test_a_merge_of_a_large_file_takes_well_under_a_second():
    """Code review of issue #31: the merge scanned every key once per shape (``startswith(k + '|')``), quadratic in
    the file's size; on the shipped M2 preset (11273 keys, 2133 shapes) it took 1.79 s, under the save's locks."""
    import time

    tune = {"__times__": {}, "__timed__": {}}
    for i in range(2200):
        k = f"{1024 + i}x512x2"
        tune[k] = ("frag", 2, 2, 1)
        tune["__times__"][k] = {"frag": 1.0}
        tune["__timed__"][k] = "0.0.1"
        for fam in ("frag", "fragh", "v3", "mlx"):
            tune[f"{k}|{fam}"] = ("mlx",)
    t = time.perf_counter()
    kernels._merge_tune(dict(tune), dict(tune))
    assert time.perf_counter() - t < 0.3


def test_a_save_that_takes_a_shape_from_the_file_drops_this_processs_family_winners(tmp_path, monkeypatch):
    """Code review of issue #31: the merge left out ``mine``'s ``key|family`` winners for a shape taken from the file,
    but the save only updated the cache in memory, so they stayed there and the next save wrote them back."""
    import json

    A = "1024x512x1"
    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_PRESET", {})
    path.write_text(json.dumps({A: ["frag", 4, 2, 1], f"{A}|frag": ["frag", 4, 2, 1], "__timed__": {A: mx.__version__}}))
    monkeypatch.setattr(kernels, "_TUNE", {A: ("mlx",), f"{A}|frag": ("frag", 2, 2, 1), f"{A}|mlx": ("mlx",),
                                           "__timed__": {A: "0.0.1"}})
    kernels._save_tune()
    assert f"{A}|mlx" not in kernels._TUNE and kernels._TUNE[f"{A}|frag"] == ("frag", 4, 2, 1)
    kernels._save_tune()
    assert f"{A}|mlx" not in json.loads(path.read_text())


def test_a_save_that_takes_a_shape_from_the_file_drops_the_cached_choice(tmp_path, monkeypatch):
    """Code review of issue #31: the save took the file's choice for a shape into the cache, but this process kept
    running the choice it had cached (``_CHOICE``, ``_DENSE``) while the cache called the shape fresh."""
    import json

    A, P = "1024x512x2", "1024x512xP"
    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_PRESET", {})
    path.write_text(json.dumps({A: ["frag", 2, 2, 1], P: ["dense"],
                                "__timed__": {A: mx.__version__, P: mx.__version__}}))
    monkeypatch.setattr(kernels, "_TUNE", {A: ("mlx",), P: ("quant",), "__timed__": {A: "0.0.1", P: "0.0.1"}})
    monkeypatch.setattr(kernels, "_CHOICE", {(1024, 512, 2, "float16"): ("mlx",), (1024, 512, 3, "float16"): ("mlx",)})
    monkeypatch.setattr(kernels, "_DENSE", {(1024, 512, "float16"): False})
    kernels._save_tune()
    assert kernels._CHOICE == {(1024, 512, 3, "float16"): ("mlx",)} and kernels._DENSE == {}


def test_a_whole_step_choice_records_the_mlx_it_was_timed_under():
    """Code review of issue #31: whole-step tuning wrote its choice with no version, so a save took the file's
    per-shape choice over it, while its row count stayed on the done list: the whole-step result was lost for good."""
    A = "1024x512x2"
    tune = {A: ("mlx",), "__timed__": {A: "0.0.1"}}
    kernels._record_whole_step(tune, A, ("frag", 2, 2, 1))
    theirs = {A: ("v3", 1), "__timed__": {A: mx.__version__}}
    assert kernels._merge_tune(tune, theirs)[A] == ["frag", 2, 2, 1]


def test_a_merge_writes_no_empty_version_map():
    """Code review of issue #31: a merge could leave ``"__timed__": {}``, which every load read as a file to
    convert, and saved again."""
    out = kernels._merge_tune({"1024x512x1": ("mlx",)}, {"__timed__": {"1024x512x1": "0.0.1"}})
    assert "__timed__" not in out and "__times__" not in out


@pytest.mark.parametrize("versions", [{"__timed__": {"1024x512x1": "0.0.1"}}, {"__mlx__": "0.0.1"}],
                         ids=["per-shape", "from-before-per-shape"])
def test_the_mlx_upgrade_line_comes_once_while_no_shape_is_timed_again(tmp_path, monkeypatch, capsys, versions):
    """Code review of issue #31: until a shape was timed under the new MLX, every start rewrote the file and printed
    the line again; with stale shapes of a model nobody serves, at every start for good."""
    import os

    A = "1024x512x1"
    path = _old_file(tmp_path, monkeypatch, {A: ["mlx"], **versions})
    kernels._load_tune()
    assert "MLX is" in capsys.readouterr().err
    os.utime(path, (0, 0))
    monkeypatch.setattr(kernels, "_TUNE", None)
    assert kernels._stale_keys(kernels._load_tune()) == [A]
    assert "MLX is" not in capsys.readouterr().err and os.path.getmtime(path) == 0


def test_a_merge_keeps_a_timing_over_a_shape_with_no_version():
    """Code review of issue #31, finding 3: neither side timed P under this MLX, and ``mine`` has no version for it
    (a preset choice): ``mine`` won, and the other process's timing of P, its choice and its version were lost."""
    P = "1024x512x2"
    mine = {P: ("mlx",)}
    theirs = {P: ("frag", 2, 2, 1), "__times__": {P: {"frag": 1.0}}, "__timed__": {P: "0.0.2"}}
    out = kernels._merge_tune(mine, theirs)
    assert (out[P], out["__times__"][P], out["__timed__"][P]) == (("frag", 2, 2, 1), {"frag": 1.0}, "0.0.2")
    assert kernels._merge_tune(theirs, mine)[P] == ("frag", 2, 2, 1)  # (a timing on this side: mine, as before)


def test_a_seed_from_before_files_recorded_an_mlx_version_is_stale(tmp_path, monkeypatch):
    """Code review of issue #31, finding 4: a seed with no ``__mlx__`` had every shape as timed under this MLX, so
    none was timed again; before per-shape versions a seed was stale whole. Its shapes keep their choices, stale."""
    import json

    _version_dir(tmp_path, monkeypatch)
    (tmp_path / kernels._TUNE_FILE).write_text(json.dumps({"1024x512x1": ["frag", 2, 2, 1]}))
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 2, 2, 1) and kernels._stale_keys(tune) == ["1024x512x1"]
