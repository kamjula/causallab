"""CUPED: use pre-period numbers to shrink the noise in an A/B test.

The idea is simple: users who spent a lot before the experiment usually
spend a lot during it too. If we pull that predictable part out first,
the leftover noise is smaller and the same lift is easier to see."""

import numpy as np


def cuped_theta(pre, post):
    # mismatched lengths used to fail deep inside numpy with a
    # confusing concatenation error. fail with a clear message instead.
    if len(pre) != len(post):
        raise ValueError("pre and post must have the same length")
    # a slope needs at least two users: with one user (or none) the
    # variance math quietly gives nan, so fail loudly instead.
    if len(pre) < 2:
        raise ValueError("cuped needs at least two users to fit a slope")
    # a nan or inf in pre or post would quietly poison the covariance
    # and return nan for the slope. fail loudly instead.
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("pre and post must not contain nan or inf")
    # how strongly the pre-period predicts the post-period:
    # the slope of post on pre. cov and var must use the same ddof,
    # otherwise the slope is off by n / (n - 1) on small samples.
    var = np.var(pre, ddof=1)
    # a flat pre-period (every user spent the same before) has no
    # signal to adjust with: dividing by zero would quietly give
    # inf/nan, so fail loudly instead.
    if var == 0:
        raise ValueError("pre-period has no variance, cuped cannot adjust")
    return np.cov(pre, post)[0, 1] / var


def adjust(pre, post):
    # subtract the predictable part of post, so only the experiment
    # effect and real noise remain
    theta = cuped_theta(pre, post)
    return post - theta * (pre - pre.mean())
