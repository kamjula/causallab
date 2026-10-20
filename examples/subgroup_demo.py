"""subgroup demo: the overall lift can hide that the treatment helps
some users a lot and others barely at all.

Build an experiment where the true lift is 8 for heavy spenders and 2
for light spenders (overall about 5), then estimate the lift per
segment and compare with the single overall number."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from simulator import estimate, run_experiment
from subgroups import estimate_by_segment


def report(name, diff, se, p):
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"{name}: lift = {diff:.2f}  (se = {se:.2f}, 95% CI: [{lo:.2f}, {hi:.2f}], "
          f"p = {p:.4f} -> {verdict})")


if __name__ == "__main__":
    # start with no lift, then add a bigger lift for the heavy spenders
    pre, post, group = run_experiment(n=4000, lift=0.0)
    heavy = pre >= np.median(pre)
    post[(group == 1) & heavy] += 8.0
    post[(group == 1) & ~heavy] += 2.0

    diff, se, p = estimate(pre, post, group)
    print("true lift: 8.0 for heavy spenders, 2.0 for light spenders\n")
    report("overall       ", diff, se, p)
    print()
    for name, d, s, pv in estimate_by_segment(pre, post, group):
        report(f"{name:14s}", d, s, pv)
    print()
    print("the overall lift misses the story: the change is worth it")
    print("for heavy spenders and barely moves light spenders")
