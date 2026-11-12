# small checks for the sequential testing math.
# run from the repo root: python -m pytest tests/ -q

from src.sequential import budget_bar, peek_significant


def test_budget_bar_splits_alpha_evenly():
    # 5% budget over 5 peeks -> each peek must clear 1%
    assert budget_bar() == 0.01
    assert budget_bar(alpha=0.05, peeks=1) == 0.05
    assert budget_bar(alpha=0.10, peeks=2) == 0.05


def test_budget_bar_rejects_bad_alpha_or_peeks():
    # zero peeks used to crash with ZeroDivisionError, and an alpha
    # outside (0, 1) quietly made the bar meaningless (0 means nothing
    # is ever significant, over 1 means everything is); both should
    # fail with a clear error
    for alpha, peeks in [(0.05, 0), (0.05, -2), (0, 5), (-0.05, 5),
                         (1.0, 5), (1.5, 5)]:
        try:
            budget_bar(alpha=alpha, peeks=peeks)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"budget_bar(alpha={alpha}, peeks={peeks}) should fail")


def test_peek_significant_uses_the_bar():
    bar = budget_bar()
    assert peek_significant(0.005, bar) is True
    assert peek_significant(0.02, bar) is False
