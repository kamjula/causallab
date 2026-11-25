# small checks for the segment split.
# run from the repo root: python -m pytest tests/ -q

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from subgroups import estimate_by_segment, make_segments


def test_make_segments_splits_evenly():
    # 6 users, 2 segments -> the 3 lowest in one, the 3 highest in the other
    pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert list(make_segments(pre, 2)) == [0, 0, 0, 1, 1, 1]


def test_make_segments_rejects_bad_counts():
    # 0 or a negative count used to hand back nonsense labels (-1, -2)
    # with no complaint, and a fractional, text or missing count failed
    # inside numpy with a confusing TypeError; all of them should fail
    # with a clear error
    pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for bad in [0, 1, -1, 1.5, "two", None, True]:
        try:
            make_segments(pre, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"make_segments(pre, {bad!r}) should fail")


def test_make_segments_rejects_bad_pre():
    # a nan in pre used to quietly put users in the wrong segment
    # (the quantile edges come out nan), a text pre failed inside
    # numpy with a confusing UFuncTypeError, and an empty or 2-d pre
    # failed with a confusing IndexError; all of them should fail
    # with a clear error
    for bad in [np.array([1.0, np.nan, 3.0, 4.0]),
                np.array([1.0, np.inf, 3.0, 4.0]),
                ["a", "b", "c", "d"],
                np.array([]),
                np.array([[1.0, 2.0], [3.0, 4.0]])]:
        try:
            make_segments(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"make_segments({bad!r}) should fail")


def test_estimate_by_segment_rejects_bad_counts():
    # the guard lives in make_segments, but estimate_by_segment is
    # the way most callers reach it, so it should fail there too
    pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    post = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    group = np.array([0, 1, 0, 1, 0, 1])
    try:
        estimate_by_segment(pre, post, group, n_segments=0)
    except ValueError:
        pass
    else:
        raise AssertionError("estimate_by_segment(n_segments=0) should fail")


def test_estimate_by_segment_works_on_lists():
    # make_segments takes lists, but estimate_by_segment used to fail
    # on them with a confusing TypeError (a boolean mask cannot
    # index a list); lists should give the same answer as arrays
    pre = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    post = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    group = [0, 1, 0, 1, 0, 1]
    got = estimate_by_segment(pre, post, group)
    want = estimate_by_segment(np.array(pre), np.array(post), np.array(group))
    assert [g[0] for g in got] == [w[0] for w in want]
    for g, w in zip(got, want):
        assert g[1] == w[1] and g[2] == w[2] and g[3] == w[3]
