"""hte demo: the treatment helps heavy spenders more than light
ones, and the lift grows smoothly with pre-period spend.

True lift per user: 2 + 0.06 * (pre - 100), so a user at pre=100
gains 2 and a user at pre=200 gains 8. The T-learner fits a model
per group and recovers that per-user pattern from the data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from simulator import estimate, run_experiment
from hte import ate, cate, fit_t_learner

if __name__ == "__main__":
    pre, post, group = run_experiment(n=4000, lift=0.0)
    true_lift = 2.0 + 0.06 * (pre - 100.0)  # bigger lift for heavier spenders
    post[group == 1] += true_lift[group == 1]

    diff, se, p = estimate(pre, post, group)
    control, treated = fit_t_learner(pre, post, group)
    pred = cate(pre, control, treated)
    ate_hat = pred.mean()

    print(f"true average lift:   {true_lift.mean():.2f}")
    print(f"T-learner ATE:       {ate_hat:.2f}")
    print(f"plain estimate:      {diff:.2f} (se = {se:.2f}, p = {p:.4f})\n")

    print("predicted lift for example users (pre spend, true lift, predicted):")
    for spend in (50, 100, 150, 200):
        idx = np.abs(pre - spend).argmin()
        print(f"  pre = {pre[idx]:5.1f}: true = {true_lift[idx]:.2f}, predicted = {pred[idx]:.2f}")

    corr = np.corrcoef(true_lift, pred)[0, 1]
    print(f"\ncorrelation between true and predicted per-user lift: {corr:.2f}")
    print("the T-learner finds who the change helps most, not just the average")
