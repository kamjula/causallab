"""Peek at an experiment early without cheating.

A normal p-value breaks if you check it every day: the more you peek,
the more likely you are to see a lucky-looking "significant" result
even when there is no real lift. The fix here is simple: decide how
many times you plan to peek, then split your 5% false-positive budget
across all the peeks. Each peek has to clear a stricter bar.

Not the fanciest sequential test out there, just the simplest one
that keeps you honest."""


def budget_bar(alpha=0.05, peeks=5):
    # zero peeks used to crash with ZeroDivisionError, and an alpha
    # outside (0, 1) quietly made the bar meaningless: 0 or less means
    # nothing is ever significant, over 1 means everything is.
    if peeks < 1:
        raise ValueError("peeks must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    # slice the error budget evenly across the planned peeks
    return alpha / peeks


def peek_significant(p, bar):
    # True only if this peek's p-value clears the stricter bar
    return p < bar
