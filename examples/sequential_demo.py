"""sequential demo: what peeking does to a plain p-value, and the simple fix.

Peek 5 times during one experiment (every 500 new users). With true
lift = 0, any "significant" flag is a false alarm. The plain 5% bar
flags way too often because 5 peeks are 5 chances to get lucky; the
stricter bar from sequential.py keeps it honest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from sequential import budget_bar
from simulator import estimate


def peek_pvalues(lift=0.0, peeks=5, batch=500):
    # build one experiment in batches and peek after every batch,
    # like checking the dashboard every few days
    rng = np.random.default_rng()
    pre, post, group = [], [], []
    ps = []
    for _ in range(peeks):
        p_batch = rng.normal(100, 25, size=batch)
        g_batch = rng.integers(0, 2, size=batch)
        post_batch = p_batch + rng.normal(0, 10, size=batch)
        post_batch[g_batch == 1] += lift
        pre.append(p_batch)
        post.append(post_batch)
        group.append(g_batch)
        _, _, p = estimate(np.concatenate(pre), np.concatenate(post),
                           np.concatenate(group))
        ps.append(p)
    return ps


if __name__ == "__main__":
    trials = 200
    bar = budget_bar(alpha=0.05, peeks=5)
    print(f"5 peeks (batches of 500 users), {trials} experiments, true lift = 0")
    print(f"stricter per-peek bar = {bar:.4f}\n")

    plain_flags = 0
    fixed_flags = 0
    for _ in range(trials):
        ps = peek_pvalues()
        if any(p < 0.05 for p in ps):
            plain_flags += 1
        if any(p < bar for p in ps):
            fixed_flags += 1

    print(f"plain 5% bar:    flagged in {plain_flags}/{trials} "
          f"({100 * plain_flags / trials:.1f}%) of experiments")
    print(f"stricter bar:    flagged in {fixed_flags}/{trials} "
          f"({100 * fixed_flags / trials:.1f}%) of experiments")
    print()
    print("peeking with the plain bar manufactures false alarms;")
    print("the stricter bar keeps them near 5%")
