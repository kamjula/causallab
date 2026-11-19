# small checks for the T-learner.
# run from the repo root: python -m pytest tests/ -q

import numpy as np

from src.hte import ate, fit_t_learner


def test_fit_t_learner_needs_both_groups():
    # an all-control or all-treated split used to fail deep inside
    # sklearn with a confusing "0 sample(s)" error; it should fail
    # with a clear message instead
    pre = np.array([10.0, 20.0, 30.0, 40.0])
    post = np.array([10.0, 20.0, 35.0, 45.0])
    for group in [np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]:
        try:
            fit_t_learner(pre, post, group)
        except ValueError as exc:
            assert "both groups" in str(exc)
        else:
            raise AssertionError("fit_t_learner should fail with one group only")
    # a normal split still works: treated users gain 5 on average
    control, treated = fit_t_learner(
        pre, post, np.array([0, 0, 1, 1]))
    assert abs(ate(pre, post, np.array([0, 0, 1, 1])) - 5.0) < 1e-9
