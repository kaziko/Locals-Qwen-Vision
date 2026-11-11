"""Issue #56: the prompt cache follows the memory macOS has available. The cache budget (GLOSSARY.md) is capped at
start by the available system memory less the memory floor, and before each admission it shrinks, or grows back up
to the start budget, by the same rule."""

import sys

import pytest

from locals import sysmem


@pytest.mark.skipif(sys.platform != "darwin", reason="kern.memorystatus_level is a macOS sysctl")
def test_available_bytes_reads_a_share_of_physical_memory():
    total = sysmem.physical_bytes()
    got = sysmem.available_bytes()
    assert total > 1 << 30
    assert isinstance(got, int) and 0 < got <= total


def test_a_failed_read_is_unknown(monkeypatch):
    monkeypatch.setattr(sysmem, "_sysctl_int", lambda name: None)
    assert sysmem.available_bytes() is None


def test_the_floor_setting(monkeypatch, capsys):
    monkeypatch.delenv("LOCALS_MEMORY_FLOOR_PCT", raising=False)
    assert sysmem.floor_pct() == 20.0
    monkeypatch.setenv("LOCALS_MEMORY_FLOOR_PCT", "0")
    assert sysmem.floor_pct() == 0.0  # off
    monkeypatch.setenv("LOCALS_MEMORY_FLOOR_PCT", "12.5")
    assert sysmem.floor_pct() == 12.5
    assert capsys.readouterr().err == ""
    for bad in ("lots", "-5", "100"):
        monkeypatch.setenv("LOCALS_MEMORY_FLOOR_PCT", bad)
        assert sysmem.floor_pct() == 20.0
        assert "LOCALS_MEMORY_FLOOR_PCT" in capsys.readouterr().err


G = 1 << 30
LIMIT = 26800603136  # max_recommended_working_set_size of a 32 GB Mac (issue #56)
FLOOR = 6871947673  # 20% of 32 GiB


def test_start_budget_without_cache_gb_is_capped_by_available_memory_but_not_below_1_5_gib():
    from locals.service import cache_budget

    # today's formula: free = 26.8 GB - 3 GiB active - 5 GiB = 18.2 GB, capped at 16 GiB
    assert cache_budget(LIMIT, 3 * G, 0) == 16 * G
    assert cache_budget(LIMIT, 3 * G, 0, available=10 * G, floor=FLOOR) == 3865470567  # 10 GiB - floor
    assert cache_budget(LIMIT, 3 * G, 8 * G, available=10 * G, floor=FLOOR) == 3865470567  # wins over the longest one
    assert cache_budget(LIMIT, 3 * G, 0, available=7 * G, floor=FLOOR) == 1610612736  # 1.5 GiB at least
    assert cache_budget(LIMIT, 3 * G, 0, available=None, floor=FLOOR) == 16 * G  # unknown: as today
    assert cache_budget(LIMIT, 3 * G, 0, available=7 * G, floor=0) == 16 * G  # floor 0: off, as today


def test_start_budget_with_cache_gb_is_exactly_that():
    from locals.service import cache_budget

    assert cache_budget(LIMIT, 3 * G, 0, cache_bytes=4 * G, available=7 * G, floor=FLOOR) == 4 * G
    assert cache_budget(LIMIT, 3 * G, 0, cache_bytes=20 * G, available=64 * G, floor=FLOOR) == 20 * G


M = 1 << 20


def _entry():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    c = ArraysCache(1)
    c[0] = mx.zeros((1, M // 4))  # float32: 1 MiB
    return [c]


def _lru(n, start):
    """A prompt cache of ``n`` 1 MiB entries, each its own conversation, made with budget ``start``."""
    from locals.hybrid_batch import chain_cache

    lru = chain_cache(max_size=256, max_bytes=start)
    for i in range(n):
        lru.insert_cache("m", [1000 + i, 1, 2], _entry(), cache_type="user")
    return lru


def _guard(start, floor, reading):
    """The engine's memory guard, with a GPU limit no admission comes near and a fake system-memory reader that
    returns ``reading[0]``."""
    from types import SimpleNamespace

    import mlx.core as mx

    from locals.hybrid_batch import HybridBatchEngine

    reads = []
    e = SimpleNamespace(counters={}, snapshots={}, memory_limit=mx.get_active_memory() + (1 << 40),
                        memory_floor=floor, start_budget=start, read_available=lambda: (reads.append(1), reading[0])[1])
    for name in ("_make_room", "_trim_for", "_pressure_trim"):
        setattr(e, name, getattr(HybridBatchEngine, name).__get__(e))
    e._kv_bytes_per_token = lambda: 0
    e._admit_bytes = lambda rows, tokens: 0
    e.reads = reads
    return e


def test_low_memory_trims_the_cache_before_an_admission_least_valuable_first():
    floor = 100 * M
    reading = [floor - 4 * M]  # 4 MiB short of the floor
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 2 * M and len(lru) == 2  # budget = min(8, 6 + (-4)) MiB
    assert lru.fetch_nearest_cache("m", [1005, 1, 2])[1] == [] and lru.fetch_nearest_cache("m", [1004, 1, 2])[1] == []
    assert lru.fetch_nearest_cache("m", [1000, 1, 2])[0] is None  # the oldest went first
    assert e.counters == {"pressure_trims": 1}  # not a GPU-limit trim: no cache_trims


def test_the_budget_grows_back_to_the_start_budget_and_not_above():
    floor = 100 * M
    reading = [floor - 4 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 2 * M
    reading[0] = floor + (1 << 30)  # memory came back
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 8 * M
    for i in range(10):
        lru.insert_cache("m", [2000 + i, 1, 2], _entry(), cache_type="user")
    assert lru.nbytes == 8 * M  # new entries kept up to the start budget
    assert e.counters == {"pressure_trims": 1}


def test_cache_gb_is_a_hard_cap_that_low_memory_still_goes_below():
    from locals.service import cache_budget

    floor = 100 * M
    start = cache_budget(LIMIT, 3 * G, 0, cache_bytes=4 * M, available=floor + G, floor=floor)
    assert start == 4 * M
    reading = [floor + (64 << 30)]
    e, lru = _guard(start, floor, reading), _lru(3, start)
    for reading[0] in (floor + (64 << 30), floor + 5 * M, floor + G):
        e._make_room(lru, tokens=16, copies=0, held=0)
        assert lru.max_bytes <= 4 * M and lru.nbytes <= 4 * M
    reading[0] = floor - 2 * M
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 1 * M and lru.max_bytes == 1 * M  # below --cache-gb


def test_floor_0_never_trims_for_memory():
    e, lru = _guard(8 * M, 0, [0]), _lru(6, 8 * M)  # nothing available, but the check is off
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 6 * M and lru.max_bytes == 8 * M and e.counters == {} and e.reads == []


def test_an_unknown_reading_neither_trims_nor_grows():
    floor = 100 * M
    reading = [floor - 4 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    e._make_room(lru, tokens=16, copies=0, held=0)
    reading[0] = None
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 2 * M and lru.nbytes == 2 * M  # no growth
    e2, lru2 = _guard(8 * M, floor, [None]), _lru(6, 8 * M)
    e2._make_room(lru2, tokens=16, copies=0, held=0)
    assert lru2.nbytes == 6 * M and lru2.max_bytes == 8 * M and e2.counters == {}  # no trim


def test_a_gpu_limit_trim_counts_only_cache_trims():
    import mlx.core as mx

    floor = 100 * M
    e, lru = _guard(8 * M, floor, [floor + (1 << 30)]), _lru(6, 8 * M)  # plenty of system memory
    e.memory_limit = mx.get_active_memory() - 3 * M  # 3 MiB over the GPU limit
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes < 6 * M
    assert e.counters == {"cache_trims": 1}


def test_stats_show_the_budget_the_start_budget_available_memory_and_pressure_trims(monkeypatch):
    from types import SimpleNamespace

    from locals.service import BatchService

    monkeypatch.setattr(sysmem, "physical_bytes", lambda: 32 * G)
    svc = BatchService.__new__(BatchService)
    svc.engine, svc.active, svc.tokens_out, svc.prompt_tokens, svc.cached_tokens = "locals", 0, 0, 0, 0
    svc._eng = SimpleNamespace(counters={"steps": 3}, memory_floor=FLOOR, start_budget=16 * G,
                               read_available=lambda: 8 * G, active=0, tokens_out=0)
    svc._lru = _lru(2, 16 * G)
    svc._lru.max_bytes = 4 * G
    s = svc.stats()
    assert s["prompt_cache_budget_mb"] == 4 * G / 1e6 and s["prompt_cache_start_mb"] == 16 * G / 1e6
    assert s["memory_available_pct"] == 25 and s["pressure_trims"] == 0
    svc._eng.read_available = lambda: None
    assert svc.stats()["memory_available_pct"] is None


def test_locals_status_prints_the_budget_and_pressure_trims(monkeypatch, capsys):
    from types import SimpleNamespace

    from locals import app

    eng = {"loaded": True, "requests": 5, "prompt_tokens": 100, "cached_tokens": 50, "completion_tokens": 10,
           "decode_s": 1.0, "forced_tokens": 0, "tool_calls": 0, "prompt_cache_mb": 2100.0, "prompt_cache_entries": 7,
           "cache_trims": 1, "deferred": 0, "prompt_cache_budget_mb": 4300.0, "prompt_cache_start_mb": 17180.0,
           "memory_available_pct": 34, "pressure_trims": 3}
    stats = {"model": "m", "uptime_s": 9.0, "requests": 5, "active": [], "engines": {"locals": eng},
             "memory": {"gpu_active_mb": 9000.0, "gpu_peak_mb": 9500.0}}
    monkeypatch.setattr(app, "server_alive", lambda port: {"status": "ok"})
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: stats)
    app.cmd_status(SimpleNamespace(port=8321))
    out = capsys.readouterr().out
    assert "prompt cache 2.1 GB in 7 entries" in out and "trimmed for memory 1x" in out
    assert "cache budget 4.3 GB now, 17.2 GB at start" in out
    assert "system memory 34% available" in out and "trimmed for system memory 3x" in out
