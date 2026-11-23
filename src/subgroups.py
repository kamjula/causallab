"""Estimate the lift separately per user segment.

The treatment might help some users more than others: a new checkout
page could win over heavy spenders while leaving light spenders
untouched. One overall lift number hides that, so split users into
segments by their pre-period spend and run the same lift estimate in
each one. No model here, just the t-test math on each slice."""

import numpy as np

from simulator import estimate


def make_segments(pre, n_segments=2):
    # n_segments is a count of slices: 0 or a negative count used to
    # hand back nonsense labels (-1, -2) with no complaint, and a
    # fractional, text or missing count failed inside numpy with a
    # confusing TypeError. fail with a clear message instead.
    if not isinstance(n_segments, (int, float)) or isinstance(n_segments, bool):
        raise ValueError("n_segments must be a whole number")
    if int(n_segments) != n_segments or n_segments < 2:
        raise ValueError("n_segments must be a whole number of 2 or more")
    # a nan in pre used to quietly put users in the wrong segment
    # (the quantile edges come out nan), a text pre failed inside
    # numpy with a confusing UFuncTypeError, and an empty pre failed
    # with a confusing IndexError. fail with a clear message instead.
    pre = np.asarray(pre)
    if pre.dtype.kind not in "iuf":
        raise ValueError("pre must be numbers")
    pre = pre.astype(float)
    if pre.ndim != 1:
        raise ValueError("pre must be one-dimensional")
    if len(pre) == 0:
        raise ValueError("pre must not be empty")
    if not np.isfinite(pre).all():
        raise ValueError("pre must not contain nan or inf")
    # split users by pre-period spend into equal-sized segments;
    # 0 = lowest spenders, n_segments - 1 = highest spenders
    edges = np.quantile(pre, np.linspace(0, 1, n_segments + 1))
    seg = np.digitize(pre, edges[1:-1])
    return np.clip(seg, 0, n_segments - 1)


def estimate_by_segment(pre, post, group, n_segments=2):
    # the same lift estimate as the simulator, computed inside each
    # segment; returns (name, lift, se, p) per segment
    if n_segments == 2:
        names = ["light spenders", "heavy spenders"]
    else:
        names = [f"segment {i}" for i in range(n_segments)]
    seg = make_segments(pre, n_segments)
    results = []
    for i in range(n_segments):
        mask = seg == i
        diff, se, p = estimate(pre[mask], post[mask], group[mask])
        results.append((names[i], diff, se, p))
    return results
