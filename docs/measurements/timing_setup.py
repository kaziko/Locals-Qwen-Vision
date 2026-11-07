"""What the timing scripts (two-token-pass/step_timing.py, mtp/mtp_test.py) share (issue #25): the hashes of the
scripts that made a result, the Locals servers running at the start and the end of a run, and a busy marker.

The busy marker makes this process count as busy for other Locals processes (kernels.others_busy): a server's idle
tuner then waits instead of timing kernels on the GPU during a measurement. It marks from a thread once a second,
as a server does, and not at all inside a timed section (``with marker.quiet():``): it marks once on the way in,
so a section shorter than the 3 s window others_busy looks back is covered, and the thread waits on a lock meanwhile.
"""
import contextlib
import hashlib
import os
import threading

from locals import kernels


def script_hashes(*paths) -> dict:
    """{file name: sha256 of its bytes}: which version of each script made a result."""
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            out[os.path.basename(p)] = hashlib.sha256(f.read()).hexdigest()
    return out


def server_pids() -> list:
    """The Locals servers running on this Mac now (any port, any install): locals.app's own check."""
    from locals.app import _locals_server_pids

    return sorted(_locals_server_pids())


class BusyMarker:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._lock = threading.Lock()  # held inside a timed section: the thread marks nothing then
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "BusyMarker":
        kernels.mark_busy(now=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="timing-busy", daemon=True)
        self._thread.start()
        return self

    def _beat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if not self._stop.is_set():
                    kernels.mark_busy(now=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @contextlib.contextmanager
    def quiet(self):
        """A timed section: marked once now, then no marks until it ends."""
        with self._lock:
            kernels.mark_busy(now=True)
            yield

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
