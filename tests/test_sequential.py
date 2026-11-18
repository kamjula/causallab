# small checks for the sequential peeking test.
# run from the repo root: python -m pytest tests/ -q

from src.sequential import budget_bar


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
