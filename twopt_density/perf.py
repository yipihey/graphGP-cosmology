"""Lightweight performance counters: wall-time timers + named counters + peak RSS.

No dependencies. Usage:

    from twopt_density import perf
    with perf.timer("measure_K2d"):
        ...
    perf.count("measure_K2d.pairs", n_pairs)      # accumulate a quantity
    @perf.timed()                                  # or decorate a function
    def f(...): ...
    perf.report()                                  # print a sorted table
    perf.reset()

``timer`` accumulates calls and total/own wall time per name; ``count``
accumulates arbitrary integer/float quantities (e.g. pair counts, N points);
``report`` prints a table sorted by total time with calls, total, mean, and any
associated counters, plus peak resident memory.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager

_TIMES: dict = {}      # name -> [n_calls, total_seconds]
_COUNTS: dict = {}     # name -> accumulated quantity


def reset():
    _TIMES.clear(); _COUNTS.clear()


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        e = _TIMES.setdefault(name, [0, 0.0])
        e[0] += 1; e[1] += dt


def timed(name: str | None = None):
    def deco(fn):
        nm = name or fn.__name__
        @functools.wraps(fn)
        def wrap(*a, **k):
            with timer(nm):
                return fn(*a, **k)
        return wrap
    return deco


def count(name: str, value=1):
    _COUNTS[name] = _COUNTS.get(name, 0) + value


def peak_rss_gb() -> float:
    import resource, sys
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # KiB (Linux), bytes (macOS)
    return m / 1024**3 if sys.platform == "darwin" else m / 1024**2


def report(title: str = "performance"):
    if not _TIMES and not _COUNTS:
        return
    print(f"\n=== {title} (peak RSS {peak_rss_gb():.1f} GiB) ===")
    print(f"{'timer':32s}{'calls':>7}{'total_s':>10}{'mean_s':>9}")
    for nm, (n, t) in sorted(_TIMES.items(), key=lambda x: -x[1][1]):
        print(f"{nm:32s}{n:7d}{t:10.2f}{t/max(n,1):9.3f}")
    if _COUNTS:
        print("counters:")
        for nm, v in sorted(_COUNTS.items()):
            print(f"  {nm:38s}{v:>16,.0f}")


def snapshot():
    """Return a copy of the current timers/counters (for programmatic use)."""
    return {k: list(v) for k, v in _TIMES.items()}, dict(_COUNTS)
