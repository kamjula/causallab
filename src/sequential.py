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
    # a text or missing p or bar used to fail inside the comparison
    # with a confusing TypeError. fail with a clear message instead.
    if not isinstance(p, (int, float)):
        raise ValueError("p must be a number")
    if not isinstance(bar, (int, float)):
        raise ValueError("bar must be a number")
    # nan or inf would quietly answer False (nothing is ever below
    # nan, and an inf p never clears a bar). fail clearly instead.
    if not math.isfinite(p) or not math.isfinite(bar):
        raise ValueError("p and bar must be finite")
    # a p-value only makes sense between 0 and 1, and the bar is a
    # slice of the same 0-1 budget: a bar of 0 would never flag
    # anything, a bar above 1 would flag everything. fail clearly instead.
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if not 0 < bar < 1:
        raise ValueError("bar must be between 0 and 1")
    # True only if this peek's p-value clears the stricter bar
    return p < bar
