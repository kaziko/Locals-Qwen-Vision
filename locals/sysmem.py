"""The memory macOS has available to every app, for the prompt cache's budget (issue #56; GLOSSARY.md "Cache budget",
"Memory floor"). ``kern.memorystatus_level`` is the percent of physical memory macOS counts as available. One ``available_bytes()`` took 1.96 us (32 GB Mac, the project venv's CPython, 10,000
calls; triage timed the bare ``sysctlbyname`` read at about 1.7 us), so it is read at every check of the memory
guard. Any failure reads as unknown (None): the caller then changes nothing."""

import ctypes
import ctypes.util
import os
import sys
from typing import Optional

_LIBC = None
_PHYSICAL: Optional[int] = None


def _sysctl_int(name: str) -> Optional[int]:
    """An integer sysctl (4 or 8 bytes), or None where it cannot be read."""
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buf = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        if _LIBC.sysctlbyname(name.encode(), ctypes.byref(buf), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        if size.value == 4:
            return buf.value & 0xFFFFFFFF
        return buf.value if size.value == 8 else None
    except (OSError, AttributeError, TypeError):
        return None


def physical_bytes() -> Optional[int]:
    """Physical memory (``hw.memsize``), read once."""
    global _PHYSICAL
    if _PHYSICAL is None:
        _PHYSICAL = _sysctl_int("hw.memsize")
    return _PHYSICAL


def available_bytes() -> Optional[int]:
    """Bytes macOS counts as available now: ``kern.memorystatus_level`` / 100 x physical memory. None: unknown."""
    level, total = _sysctl_int("kern.memorystatus_level"), physical_bytes()
    if level is None or not total or not 0 <= level <= 100:
        return None
    return total * level // 100


FLOOR_PCT = 20.0  # a guess, not a measurement: the manual check of issue #56 tunes it


def floor_pct() -> float:
    """The memory floor in percent of physical memory: ``LOCALS_MEMORY_FLOOR_PCT``, else 20. 0 turns the pressure
    check off. A value that is not a number from 0 up to (not including) 100 prints a warning and gives 20."""
    v = os.environ.get("LOCALS_MEMORY_FLOOR_PCT")
    if v in (None, ""):
        return FLOOR_PCT
    try:
        pct = float(v)
    except ValueError:
        pct = -1.0
    if not 0 <= pct < 100:
        print(f"locals: LOCALS_MEMORY_FLOOR_PCT={v!r} is not a number from 0 to under 100; {FLOOR_PCT:g}%", file=sys.stderr)
        return FLOOR_PCT
    return pct


def floor_bytes() -> int:
    """The memory floor in bytes: ``floor_pct()`` of physical memory. 0 (off) when either is 0 or unknown."""
    total, pct = physical_bytes(), floor_pct()
    return int(total * pct / 100) if total and pct else 0
