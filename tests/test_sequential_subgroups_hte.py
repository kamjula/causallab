# checks for sequential peeking, per-segment lift and the T-learner.
# run from the repo root: python -m pytest tests/ -q

import sys
from pathlib import Path

import numpy as np
import pytest

# these modules import each other by plain name (from simulator import ...),
# the same way the examples load them, so put src/ on the path first
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hte import ate, cate, fit_t_learner  # noqa: E402
from sequential import budget_bar, peek_significant  # noqa: E402
from simulator import estimate  # noqa: E402
from subgroups import estimate_by_segment, make_segments  # noqa: E402


def users(rng, n, lift=0.0):
    # same shape of data as the simulator, but with a local seed so
    # every test is reproducible on its own
    pre = rng.normal(100, 25, size=n)
    group = rng.integers(0, 2, size=n)
    post = pre + rng.normal(0, 10, size=n)
    post[group == 1] += lift
    return pre, post, group


# sequential peeking

def test_budget_bar_splits_alpha_evenly():
    assert budget_bar(alpha=0.05, peeks=5) == pytest.approx(0.01)
    assert budget_bar(alpha=0.05, peeks=1) == pytest.approx(0.05)


def test_peek_must_clear_the_stricter_bar():
    bar = budget_bar(0.05, 5)
    assert peek_significant(0.009, bar)
    assert not peek_significant(0.01, bar)
    assert not peek_significant(0.04, bar)


def test_split_bar_keeps_false_alarms_near_five_percent():
    # no real lift and 5 peeks per experiment: the plain 5% bar flags far
    # more than 5% of experiments, the split bar stays close to 5%
    rng = np.random.default_rng(0)
    trials, peeks, batch = 300, 5, 200
    bar = budget_bar(0.05, peeks)
    plain = split = 0
    for _ in range(trials):
        pre, post, group = users(rng, peeks * batch)
        ps = [estimate(pre[:k], post[:k], group[:k])[2]
              for k in range(batch, peeks * batch + 1, batch)]
        plain += any(p < 0.05 for p in ps)
        split += any(peek_significant(p, bar) for p in ps)
    assert plain / trials > 0.10
    assert split / trials <= 0.08


# per-segment lift

def test_segments_are_equal_sized_and_ordered_by_spend():
    rng = np.random.default_rng(1)
    pre = rng.normal(100, 25, size=1000)
    seg = make_segments(pre, n_segments=4)
    assert set(np.unique(seg)) == {0, 1, 2, 3}
    counts = np.bincount(seg)
    assert counts.min() >= 240 and counts.max() <= 260
    means = [pre[seg == i].mean() for i in range(4)]
    assert means == sorted(means)


def test_segment_lift_shows_up_only_where_it_exists():
    # the change helps heavy spenders by 8 and does nothing for light ones
    rng = np.random.default_rng(2)
    pre, post, group = users(rng, 4000)
    heavy = pre >= np.median(pre)
    post[(group == 1) & heavy] += 8.0
    (light, light_lift, light_se, _), (heavy_name, heavy_lift, heavy_se, heavy_p) = \
        estimate_by_segment(pre, post, group, n_segments=2)
    assert (light, heavy_name) == ("light spenders", "heavy spenders")
    # each estimate should sit within 3 standard errors of the truth
    assert abs(light_lift - 0.0) < 3 * light_se
    assert abs(heavy_lift - 8.0) < 3 * heavy_se
    assert heavy_p < 0.001


# T-learner

def test_t_learner_recovers_how_lift_grows_with_spend():
    # true lift per user: 2 + 0.06 * (pre - 100)
    rng = np.random.default_rng(3)
    pre, post, group = users(rng, 4000)
    true_lift = 2.0 + 0.06 * (pre - 100.0)
    post[group == 1] += true_lift[group == 1]
    control, treated = fit_t_learner(pre, post, group)
    low, high = cate(np.array([50.0, 150.0]), control, treated)
    assert (high - low) / 100.0 == pytest.approx(0.06, abs=0.04)
    assert ate(pre, post, group) == pytest.approx(true_lift.mean(), abs=1.0)


def test_t_learner_finds_no_effect_when_there_is_none():
    rng = np.random.default_rng(4)
    pre, post, group = users(rng, 4000)
    assert abs(ate(pre, post, group)) < 1.0
