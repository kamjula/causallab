"""A/B test simulator. Makes fake users, splits them in two groups, measures the lift."""

import math
from math import erf

import numpy as np

rng = np.random.default_rng(7)


def make_users(n=2000):
    # zero or negative users makes no sense: it would silently hand
    # back empty arrays, and estimate() on those returns nan.
    # fail loudly instead.
    if n <= 0:
        raise ValueError("n must be positive")
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
    # mismatched lengths used to fail deep inside numpy with a
    # confusing boolean-index error. fail with a clear message instead.
    if len(post) != len(group):
        raise ValueError("post and group must have the same length")
    ctrl = post[group == 0].mean()
    treat = post[group == 1].mean()
    diff = treat - ctrl
    # standard error of the difference, so we see how noisy the number is
    se = np.sqrt(post[group == 0].var() / (group == 0).sum() +
                 post[group == 1].var() / (group == 1).sum())
    # no variation in post at all means the t-test has nothing to
    # work with: differing constant groups would quietly claim p = 0,
    # identical ones would quietly return nan. fail loudly instead.
    if se == 0:
        raise ValueError("post has no variation, t-test cannot run")
    # two-sided p-value from a t-test; normal approximation is fine for
    # thousands of users
    z = abs(diff / se)
    p = 1 - erf(z / np.sqrt(2))
    return diff, se, p


def sample_size(lift, sigma):
    # a lift of zero (or less) makes no sense here: you can't size an
    # experiment to detect nothing, and a negative lift would quietly
    # give the same answer as a positive one. fail loudly instead.
    if lift <= 0:
        raise ValueError("lift must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # how many users per group to spot this lift: standard two-sample
    # formula, 80% power at 5% level (z = 1.96 for alpha, 0.84 for beta)
    n = 2 * sigma ** 2 * (1.96 + 0.84) ** 2 / lift ** 2
    return math.ceil(n)


if __name__ == "__main__":
    for i in range(3):
        pre, post, group = run_experiment()
        diff, se, p = estimate(pre, post, group)
        lo, hi = diff - 1.96 * se, diff + 1.96 * se  # 95% confidence interval
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"run {i + 1}: measured lift = {diff:.2f}  (se = {se:.2f}, true lift = 5.0)")
        print(f"         95% CI: [{lo:.2f}, {hi:.2f}]")
        print(f"         p-value = {p:.4f} -> {verdict} at 5%")

    # power check: run a small pilot, measure the noise, see how many users
    # each group needs to detect a few different lifts
    print()
    _, pilot, _ = run_experiment(n=1000, lift=0.0)
    sigma = pilot.std()
    for want in (2.0, 5.0, 10.0):
        need = sample_size(want, sigma)
        print(f"to detect lift {want}: {need} users per group")
