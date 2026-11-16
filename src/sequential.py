"""Peek at an experiment early without cheating.

A normal p-value breaks if you check it every day: the more you peek,
the more likely you are to see a lucky-looking "significant" result
even when there is no real lift. The fix here is simple: decide how
many times you plan to peek, then split your 5% false-positive budget
across all the peeks. Each peek has to clear a stricter bar.

Not the fanciest sequential test out there, just the simplest one
that keeps you honest."""


def budget_bar(alpha=0.05, peeks=5):
    # slice the error budget evenly across the planned peeks
    return alpha / peeks


def peek_significant(p, bar):
    # True only if this peek's p-value clears the stricter bar
    return p < bar
