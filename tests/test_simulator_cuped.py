# small checks for the simulator and cuped math.
# run from the repo root: python -m pytest tests/ -q

import numpy as np

from src.cuped import adjust, cuped_theta
from src.simulator import estimate, make_users, run_experiment, sample_size


def test_estimate_diff_is_treatment_minus_control():
    # control around 100, treatment around 105 -> lift should be 5
    ctrl = np.array([90.0, 100.0, 110.0])
    treat = np.array([95.0, 105.0, 115.0])
    pre = np.zeros(6)
    post = np.concatenate([ctrl, treat])
    group = np.array([0, 0, 0, 1, 1, 1])
    diff, se, p = estimate(pre, post, group)
    assert diff == 5.0
    assert se > 0
    assert 0 <= p <= 1


def test_estimate_no_lift_gives_p_value_one():
    # identical groups -> no difference at all
    post = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 110.0])
    group = np.array([0, 0, 0, 1, 1, 1])
    diff, _, p = estimate(np.zeros(6), post, group)
    assert diff == 0.0
    assert p == 1.0


def test_sample_size_known_number():
    # 2 * 25^2 * (1.96 + 0.84)^2 / 5^2 = 392 exactly
    assert sample_size(5.0, 25.0) == 392


def test_sample_size_rejects_zero_or_negative_lift():
    # zero used to crash with ZeroDivisionError, negative silently gave
    # the same answer as positive; both should fail with a clear error
    for lift, sigma in [(0, 25.0), (-5.0, 25.0), (5.0, 0)]:
        try:
            sample_size(lift, sigma)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sample_size({lift}, {sigma}) should fail")


def test_sample_size_smaller_lift_needs_more_users():
    assert sample_size(2.0, 25.0) > sample_size(5.0, 25.0)
    assert sample_size(5.0, 25.0) > sample_size(10.0, 25.0)


def test_make_users_rejects_zero_or_negative_n():
    # zero used to hand back empty arrays, and estimate() on those
    # quietly returned nan; both should fail with a clear error
    for n in (0, -10):
        try:
            make_users(n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"make_users({n}) should fail")


def test_run_experiment_recovers_true_lift():
    # 4000 users, true lift 5: measured lift should be close, not exact
    _, post, group = run_experiment(n=4000, lift=5.0)
    diff, se, p = estimate(np.zeros(4000), post, group)
    assert abs(diff - 5.0) < 3.0
    assert se > 0
    assert p < 0.05
    # split should be roughly even
    assert 1800 < (group == 1).sum() < 2200


def test_cuped_theta_on_perfect_line():
    # post is exactly twice pre -> slope is 2
    pre = np.array([0.0, 1.0, 2.0, 3.0])
    post = 2 * pre
    assert abs(cuped_theta(pre, post) - 2.0) < 1e-9


def test_cuped_theta_rejects_fewer_than_two_users():
    # one user (or none) used to return nan with only a warning;
    # a slope cannot be fit from a single point, so fail clearly
    for pre, post in [(np.array([100.0]), np.array([110.0])),
                       (np.array([]), np.array([]))]:
        try:
            cuped_theta(pre, post)
        except ValueError:
            pass
        else:
            raise AssertionError("cuped_theta on <2 users should fail")


def test_cuped_adjust_removes_predictable_part():
    # post = 2 * pre + 10: after adjusting, nothing varies any more
    pre = np.array([0.0, 1.0, 2.0, 3.0])
    post = 2 * pre + 10.0
    adj = adjust(pre, post)
    assert adj.std() < 1e-9


def test_cuped_shrinks_variance_on_correlated_data():
    # pre predicts post well, so the adjusted numbers vary less
    rng = np.random.default_rng(0)
    pre = rng.normal(100, 25, size=2000)
    post = pre + rng.normal(0, 10, size=2000)
    adj = adjust(pre, post)
    assert adj.var() < post.var()
    assert adj.var() < 0.5 * post.var()


def test_estimate_rejects_mismatched_lengths():
    # post shorter than group used to fail inside numpy with a
    # confusing boolean-index error; it should fail with a clear error
    try:
        estimate(np.zeros(6), np.array([100.0, 101.0, 102.0]),
                 np.array([0, 0, 0, 1, 1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("estimate on mismatched lengths should fail")


def test_estimate_rejects_zero_variation_post():
    # every post value the same: differing constant groups used to
    # claim p = 0 with se = 0, identical ones returned nan silently;
    # a t-test needs variation, so both should fail with a clear error
    group = np.array([0, 0, 0, 1, 1, 1])
    for post in [np.array([100.0, 100.0, 100.0, 105.0, 105.0, 105.0]),
                 np.full(6, 100.0)]:
        try:
            estimate(np.zeros(6), post, group)
        except ValueError:
            pass
        else:
            raise AssertionError("estimate on constant post should fail")


def test_estimate_rejects_other_group_values():
    # a third group value used to be silently dropped from both means
    # (those users just vanished from the lift); it should fail clearly
    post = np.array([90.0, 100.0, 110.0, 95.0, 105.0, 115.0])
    group = np.array([0, 0, 2, 1, 1, 1])
    try:
        estimate(np.zeros(6), post, group)
    except ValueError:
        pass
    else:
        raise AssertionError("estimate with group value 2 should fail")
