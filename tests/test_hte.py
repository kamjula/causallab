# small checks for the T-learner.
# run from the repo root: python -m pytest tests/ -q

import numpy as np

from src.hte import ate, cate, fit_t_learner


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


def test_cate_needs_clean_pre():
    # a list pre used to fail with a confusing AttributeError, a 2-d
    # pre was silently flattened, and a nan or inf pre failed deep
    # inside sklearn with a long message; all should fail with a
    # clear message instead
    pre = np.array([10.0, 20.0, 30.0, 40.0])
    post = np.array([10.0, 20.0, 35.0, 45.0])
    group = np.array([0, 0, 1, 1])
    control, treated = fit_t_learner(pre, post, group)
    # a normal pre still works: every user gains 5
    assert (cate(pre, control, treated) == 5.0).all()
    for bad in [["a", "b", "c", "d"],
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                np.array([10.0, np.nan, 30.0, 40.0]),
                np.array([10.0, np.inf, 30.0, 40.0]),
                None]:
        try:
            cate(bad, control, treated)
        except ValueError as exc:
            assert "pre" in str(exc)
        else:
            raise AssertionError("cate should fail on a bad pre")
