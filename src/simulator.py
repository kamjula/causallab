"""A/B test simulator. Makes fake users, splits them in two groups, measures the lift."""

from math import erf

import numpy as np

rng = np.random.default_rng(7)


def make_users(n=2000):
    # each user has a natural spend level (their "type"), from the pre period
    natural = rng.normal(100, 25, size=n)
    # random split: 0 = control (old page), 1 = treatment (new page)
    group = rng.integers(0, 2, size=n)
    return natural, group


def run_experiment(n=2000, lift=5.0):
    pre, group = make_users(n)
    post = pre + rng.normal(0, 10, size=n)  # day to day noise
    post[group == 1] += lift               # the change adds this much
    return pre, post, group


def estimate(pre, post, group):
    ctrl = post[group == 0].mean()
    treat = post[group == 1].mean()
    diff = treat - ctrl
    # standard error of the difference, so we see how noisy the number is
    se = np.sqrt(post[group == 0].var() / (group == 0).sum() +
                 post[group == 1].var() / (group == 1).sum())
    # two-sided p-value from a t-test; normal approximation is fine for
    # thousands of users
    z = abs(diff / se)
    p = 1 - erf(z / np.sqrt(2))
    return diff, se, p


if __name__ == "__main__":
    for i in range(3):
        pre, post, group = run_experiment()
        diff, se, p = estimate(pre, post, group)
        lo, hi = diff - 1.96 * se, diff + 1.96 * se  # 95% confidence interval
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"run {i + 1}: measured lift = {diff:.2f}  (se = {se:.2f}, true lift = 5.0)")
        print(f"         95% CI: [{lo:.2f}, {hi:.2f}]")
        print(f"         p-value = {p:.4f} -> {verdict} at 5%")
