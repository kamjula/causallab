"""Estimate the lift separately per user segment.

The treatment might help some users more than others: a new checkout
page could win over heavy spenders while leaving light spenders
untouched. One overall lift number hides that, so split users into
segments by their pre-period spend and run the same lift estimate in
each one. No model here, just the t-test math on each slice."""

import numpy as np

from simulator import estimate


def make_segments(pre, n_segments=2):
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
