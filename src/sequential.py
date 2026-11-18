"""Peek at an experiment early without cheating.

A normal p-value breaks if you check it every day: the more you peek,
the more likely you are to see a lucky-looking "significant" result
even when there is no real lift. The fix here is simple: decide how
many times you plan to peek, then split your 5% false-positive budget
across all the peeks. Each peek has to clear a stricter bar.

Not the fanciest sequential test out there, just the simplest one
that keeps you honest."""

import math


def budget_bar(alpha=0.05, peeks=5):
    # a text or missing alpha or peeks used to fail inside the
    # division with a confusing TypeError. fail with a clear message instead.
    if not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number")
    if not isinstance(peeks, (int, float)):
        raise ValueError("peeks must be a number")
    # inf or nan would give a nonsense bar. fail clearly instead.
    if not math.isfinite(alpha) or not math.isfinite(peeks):
        raise ValueError("alpha and peeks must be finite")
    # alpha is a false-positive budget, so it has to sit between 0 and 1:
    # 0 means never significant, above 1 the budget is meaningless.
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    # peeks is a count of planned checks: zero used to crash with
    # ZeroDivisionError, and a negative or fractional count quietly
    # gave a nonsense bar. fail clearly instead.
    if peeks < 1 or int(peeks) != peeks:
        raise ValueError("peeks must be a whole number of 1 or more")
    # slice the error budget evenly across the planned peeks
    return alpha / peeks


def peek_significant(p, bar):
    # True only if this peek's p-value clears the stricter bar
    return p < bar
