"""Issue #56: the prompt cache follows the memory macOS has available. The start budget is the plain formula (or
--cache-gb; issue #58: system memory does not change it), and at each check of the memory guard the cache budget
shrinks below it, or grows back up to it, by the available memory less the memory floor."""

import sys
import time

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


def test_start_budget_without_cache_gb_is_the_formula_whatever_system_memory_reads(monkeypatch):
    """Issue #58: the start budget is what the GPU limit leaves free at start, 1.5 GiB to 16 GiB; available system
    memory and the floor do not change it (the guard caps the budget at run time, and it grows back to this)."""
    from locals.service import cache_budget

    monkeypatch.setattr(sysmem, "available_bytes", lambda: 1 * G)  # far below any floor
    # free = 26.8 GB - 3 GiB active - 5 GiB = 18.2 GB, capped at 16 GiB
    assert cache_budget(LIMIT, 3 * G, 0) == 16 * G
    # below the 16 GiB cap: free = 26.8 GB - 12 GiB - 5 GiB
    assert cache_budget(LIMIT, 12 * G, 0) == 8546992128
    assert cache_budget(LIMIT, 12 * G, 9 * G) == 9 * G  # one conversation as long as fits
    assert cache_budget(LIMIT, 12 * G, 20 * G) == 16 * G  # (still at most 16 GiB)
    assert cache_budget(LIMIT, 22 * G, 0) == 1610612736  # 1.5 GiB at least


def test_the_floor_in_bytes_follows_the_setting(monkeypatch):
    monkeypatch.setattr(sysmem, "physical_bytes", lambda: 32 * G)
    monkeypatch.delenv("LOCALS_MEMORY_FLOOR_PCT", raising=False)
    assert sysmem.floor_bytes() == FLOOR
    monkeypatch.setenv("LOCALS_MEMORY_FLOOR_PCT", "0")
    assert sysmem.floor_bytes() == 0  # off: the engine never reads system memory
    monkeypatch.delenv("LOCALS_MEMORY_FLOOR_PCT")
    monkeypatch.setattr(sysmem, "physical_bytes", lambda: None)
    assert sysmem.floor_bytes() == 0  # physical memory unknown: off


def test_start_budget_with_cache_gb_is_exactly_that():
    from locals.service import cache_budget

    assert cache_budget(LIMIT, 3 * G, 0, cache_bytes=4 * G) == 4 * G
    assert cache_budget(LIMIT, 3 * G, 0, cache_bytes=20 * G) == 20 * G


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

    def read():
        reads.append(1)
        return reading[0]

    e = SimpleNamespace(counters={}, snapshots={}, memory_limit=mx.get_active_memory() + (1 << 40),
                        memory_floor=floor, start_budget=start, read_available=read, _unseen=None, _now=time.monotonic, _freed_at=None)
    for name in ("_make_room", "_trim_for", "_pressure_trim", "_budget", "_seen", "_note_freed", "_free_unused",
                 "_room_for_copy", "_growth_trim"):
        if hasattr(HybridBatchEngine, name):  # (a missing one fails where it is called)
            setattr(e, name, getattr(HybridBatchEngine, name).__get__(e))
    e._kv_bytes_per_token = lambda: 0
    e._admit_bytes = lambda rows, tokens: 0
    e.reads = reads
    _settle()  # (what earlier tests left, freed by the guard's first step, would count as memory given back)
    return e


def _settle():
    """Free what earlier tests left (reference cycles, MLX's pool of freed buffers), as the guard does before it
    drops entries: freed here, it would count as memory the guard gave back."""
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def test_low_memory_trims_the_cache_before_an_admission_least_valuable_first():
    floor = 100 * M
    reading = [floor - 4 * M]  # 4 MiB short of the floor
    e = _guard(8 * M, floor, reading)
    lru = _lru(0, 8 * M)
    lru.insert_cache("m", [7, 7, 7], _entry(), cache_type="system")  # the oldest entry, but a system prompt
    for i in range(5):
        lru.insert_cache("m", [1000 + i, 1, 2], _entry(), cache_type="user")
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 2 * M and len(lru) == 2  # budget = min(8, 6 + (-4)) MiB
    assert lru.fetch_nearest_cache("m", [7, 7, 7])[1] == []  # the system prompt stays
    assert lru.fetch_nearest_cache("m", [1004, 1, 2])[1] == []  # and the newest checkpoint
    assert lru.fetch_nearest_cache("m", [1003, 1, 2])[0] is None
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
    start = cache_budget(LIMIT, 3 * G, 0, cache_bytes=4 * M)
    assert start == 4 * M
    reading = [floor + (64 << 30)]
    e, lru = _guard(start, floor, reading), _lru(3, start)
    for reading[0] in (floor + (64 << 30), floor + 5 * M, floor + G):
        e._make_room(lru, tokens=16, copies=0, held=0)
        assert lru.max_bytes <= 4 * M and lru.nbytes <= 4 * M
    reading[0] = floor - 2 * M
    _settle()
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
    _settle()
    e._make_room(lru, tokens=16, copies=0, held=0)
    reading[0] = None
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 2 * M and lru.nbytes == 2 * M  # no growth
    e2, lru2 = _guard(8 * M, floor, [None]), _lru(6, 8 * M)
    e2._make_room(lru2, tokens=16, copies=0, held=0)
    assert lru2.nbytes == 6 * M and lru2.max_bytes == 8 * M and e2.counters == {}  # no trim


def test_a_gpu_limit_trim_counts_only_cache_trims():
    import gc

    import mlx.core as mx

    floor = 100 * M
    e, lru = _guard(8 * M, floor, [floor + (1 << 30)]), _lru(6, 8 * M)  # plenty of system memory
    gc.collect()  # as the guard does first: arrays an earlier test left would be freed there and end the overshoot
    mx.clear_cache()
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


def test_the_same_low_reading_cuts_the_shortfall_once():
    """A reading that does not show a trim yet (it lags, or the dropped entries shared their buffers with a running
    row and freed nothing) must not cut the same shortfall again at every check: one admission runs several."""
    floor = 100 * M
    reading = [floor - 4 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    _settle()
    for _ in range(3):
        e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 2 * M and lru.max_bytes == 2 * M and e.counters == {"pressure_trims": 1}
    reading[0] = floor - 5 * M  # another app took 1 MiB more: only that is cut
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 1 * M and e.counters == {"pressure_trims": 2}
    reading[0] = floor  # the reading caught up with the 5 MiB given back, and gained no more
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 1 * M and lru.max_bytes == 1 * M and e.counters == {"pressure_trims": 2}


def test_memory_given_back_that_no_reading_shows_counts_only_for_a_while():
    """macOS's reading can take in what the guard gives back and never rise by it (another app took the memory, or
    macOS used it for something else): the guard then trusts the reading again after UNSEEN_S seconds, and does not
    stay blind below the floor (issue #58 review)."""
    from locals.hybrid_batch import UNSEEN_S

    floor = 100 * M
    reading = [floor - 4 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    now = [0.0]
    e._now = lambda: now[0]
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 2 * M and e.counters == {"pressure_trims": 1}
    now[0] = UNSEEN_S / 2  # the same reading soon after: the 4 MiB given back still count
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 2 * M and e.counters == {"pressure_trims": 1}
    now[0] = UNSEEN_S + 0.01  # still the same reading: the shortfall is real, and the rest of the cache goes
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 0 and lru.max_bytes == 0 and e.counters == {"pressure_trims": 2}


def test_checks_with_nothing_new_to_free_run_the_collector_once_a_step(monkeypatch):
    """Below the floor with the prompt cache empty, an admission's checks (and a GPU-limit trim after the floor check
    in the same call) found nothing to free but ran gc.collect() and emptied MLX's pool each time (issue #58 review):
    once per step count is enough while MLX holds no more than after the last one."""
    import gc

    import mlx.core as mx

    floor = 100 * M
    e, lru = _guard(8 * M, floor, [floor - 4 * M]), _lru(0, 8 * M)
    runs = []
    monkeypatch.setattr(gc, "collect", lambda *a: runs.append(1))
    e._make_room(lru, tokens=16, copies=0, held=0)
    e._room_for_copy(lru, 16)
    e.memory_limit = mx.get_active_memory() - 1  # over the GPU limit too: _trim_for frees for it after the floor check
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert len(runs) == 1
    e.counters["steps"] = 32  # the next growth check
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert len(runs) == 2
    keep = mx.zeros((M // 4,))  # MLX holds more than after the last one: it may be garbage, so the collector runs
    mx.eval(keep)
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert len(runs) == 3


def test_the_buffer_pool_and_snapshots_go_before_prompt_cache_entries(monkeypatch):
    import mlx.core as mx

    floor = 100 * M
    reading = [floor - 4 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    clears, seen = [], [lru]

    def clear_cache():  # MLX's pool given back to macOS: the reading shows it
        clears.append(len(seen[0]))
        reading[0] += 2 * M

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    e._make_room(lru, tokens=16, copies=0, held=0)  # 2 MiB short after the pool: the snapshots go too
    assert clears == [6, 4] and lru.nbytes == 4 * M and e.counters == {"pressure_trims": 1}  # (no snapshots)
    reading[0] = floor - 4 * M
    e.snapshots = {(1, 2): ["kv"]}
    lru2 = seen[0] = _lru(6, 8 * M)
    e._unseen = None
    clears.clear()
    e._make_room(lru2, tokens=16, copies=0, held=0)
    assert e.snapshots == {} and clears == [6, 6]  # the pool, then the snapshots: no entry dropped
    assert lru2.nbytes == 6 * M and lru2.max_bytes == 6 * M and e.counters == {"pressure_trims": 1}


def test_an_entry_moved_into_the_cache_is_not_pushed_out_by_its_own_bytes():
    """A lone row's retire moves its buffers into the prompt cache as they are: the last reading counted them as used
    memory already, so the budget makes room for them instead of dropping older entries (or the new one)."""
    floor = 100 * M
    reading = [floor]  # no room beyond the cache now
    e, lru = _guard(16 * M, floor, reading), _lru(6, 16 * M)
    _settle()
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 6 * M
    moved = _entry() + _entry()
    e._pressure_trim(lru, adding=sum(c.nbytes for c in moved))
    lru.insert_cache("m", [3000, 1, 2], moved, cache_type="assistant")
    assert lru.nbytes == 8 * M and len(lru) == 7 and e.counters == {}
    assert lru.fetch_nearest_cache("m", [1000, 1, 2])[1] == []  # the oldest entry stays
    reading[0] = floor - 1 * M  # short by 1 MiB: room for the moved entry comes out of the others
    e._pressure_trim(lru, adding=2 * M)
    lru.insert_cache("m", [3001, 1, 2], _entry() + _entry(), cache_type="assistant")
    assert lru.nbytes == 9 * M and lru.fetch_nearest_cache("m", [3001, 1, 2])[1] == []
    assert e.counters == {"pressure_trims": 1}


def test_the_system_memory_check_runs_without_a_gpu_limit():
    floor = 100 * M
    reading = [floor - 1 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    e.memory_limit = 0  # no GPU guard
    _settle()
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.nbytes == 5 * M
    reading[0] -= 1 * M
    assert e._room_for_copy(lru, 16) is True
    assert lru.nbytes == 4 * M
    reading[0] -= 1 * M
    e._serve_lru, e._next_trim, e.counters["steps"] = lru, 0, 32
    e._growth_trim(1)
    assert lru.nbytes == 3 * M and e.counters["pressure_trims"] == 3


def test_growth_trim_off_still_checks_system_memory_every_32_steps(monkeypatch):
    """Issue #58: LOCALS_GROWTH_TRIM=0 turns off only the room made under the GPU limit for the rows' KV growth; the
    memory-floor check still runs every 32 steps, so a long reply with no new admission gives memory back."""
    import mlx.core as mx

    from locals import hybrid_batch

    monkeypatch.setattr(hybrid_batch, "_GROWTH_TRIM", False)
    floor = 100 * M
    reading = [floor - 1 * M]
    e, lru = _guard(8 * M, floor, reading), _lru(6, 8 * M)
    _settle()
    e.memory_limit = mx.get_active_memory() - 3 * M  # 3 MiB over the GPU limit: a growth trim would cut for it
    e._serve_lru, e._next_trim, e.counters["steps"] = lru, 0, 32
    e._growth_trim(1)
    assert lru.nbytes == 5 * M and e.counters == {"steps": 32, "pressure_trims": 1}  # no cache_trims: no GPU-limit trim
    reading[0] -= 1 * M
    e.counters["steps"] = 40  # before the next 32-step mark: no check
    e._growth_trim(1)
    assert lru.nbytes == 5 * M
    e.counters["steps"] = 64
    e._growth_trim(1)
    assert lru.nbytes == 4 * M and e.counters == {"steps": 64, "pressure_trims": 2}


def test_a_server_started_in_low_memory_is_capped_at_the_first_check_and_grows_back_to_the_formula():
    """Issue #58: a server that starts while macOS has less than start budget + floor available keeps the formula's
    start budget; the first check of the guard caps the cache, and once memory comes back the budget is the start
    budget again (with the start-time cap of #56 it stayed at the low start for the whole process)."""
    from locals.service import cache_budget

    floor = 100 * M
    reading = [floor + 2 * M]  # far below start budget + floor
    start = cache_budget(LIMIT, 3 * G, 0)
    assert start == 16 * G
    e, lru = _guard(start, floor, reading), _lru(6, start)
    _settle()
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == 8 * M and lru.nbytes == 6 * M  # 6 MiB in the cache + 2 MiB available above the floor
    reading[0] = floor + start + G  # memory came back
    e._make_room(lru, tokens=16, copies=0, held=0)
    assert lru.max_bytes == start
