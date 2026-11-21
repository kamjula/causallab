# small checks for the sequential peeking test.
# run from the repo root: python -m pytest tests/ -q

from src.sequential import budget_bar, peek_significant


def test_budget_bar_default_split():
    # 5% budget over 5 peeks -> each peek must clear 1%
    assert budget_bar() == 0.01
    assert budget_bar(alpha=0.05, peeks=10) == 0.005


def test_budget_bar_rejects_bad_inputs():
    # zero peeks used to crash with ZeroDivisionError, a negative or
    # fractional count quietly gave a nonsense bar, and a text or
    # missing input failed inside the division with a confusing
    # TypeError; all of them should fail with a clear error
    for alpha, peeks in [(0.05, 0), (0.05, -2), (0.05, 2.5),
                         (0.05, "5"), (0.05, None),
                         (0, 5), (1.5, 5), ("0.05", 5), (None, 5),
                         (float("nan"), 5), (0.05, float("inf"))]:
        try:
            budget_bar(alpha, peeks)
        except ValueError:
            pass
        else:
            raise AssertionError(f"budget_bar({alpha!r}, {peeks!r}) should fail")


def test_peek_significant_normal_cases():
    # a peek clears the bar only when its p-value is under it
    assert peek_significant(0.005, 0.01)
    assert not peek_significant(0.02, 0.01)
    assert not peek_significant(0.01, 0.01)  # equal does not clear it


def test_peek_significant_rejects_bad_inputs():
    # a nan or inf p used to quietly answer False, a text or missing
    # p or bar failed inside the comparison with a confusing TypeError,
    # and a bar of 0 or above 1 gave a nonsense answer; all of them
    # should fail with a clear error
    for p, bar in [("0.005", 0.01), (None, 0.01), (float("nan"), 0.01),
                   (float("inf"), 0.01), (1.5, 0.01), (-0.1, 0.01),
                   (0.005, "0.01"), (0.005, None), (0.005, float("nan")),
                   (0.005, 0), (0.005, 1), (0.005, 1.5)]:
        try:
            peek_significant(p, bar)
        except ValueError:
            pass
        else:
            raise AssertionError(f"peek_significant({p!r}, {bar!r}) should fail")
