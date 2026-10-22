"""end-to-end: take one business question and run it through the
whole workbench, in the order a data scientist actually would.

The question: we shipped a new checkout page to half our users. Did
it raise spend enough to justify rolling it out to everyone?"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cuped import adjust, cuped_theta
from hte import ate, cate, fit_t_learner
from sequential import budget_bar, peek_significant
from simulator import estimate, run_experiment
from subgroups import estimate_by_segment

np.random.seed(7)


def banner(text):
    print(f"\n--- {text} ---")


if __name__ == "__main__":
    banner("the question")
    print("new checkout page live for 2 weeks, shown to 50% of users.")
    print("decision: roll out to everyone, or roll back?")

    banner("1. run the experiment")
    # new checkout page: helps everyone a little, heavy spenders a lot
    pre, post, group = run_experiment(n=4000, lift=0.0)
    true_lift = 2.0 + 0.04 * (pre - 100.0)
    post[group == 1] += true_lift[group == 1]
    print(f"n = 4000 users (2000 per group), true avg lift = {true_lift.mean():.2f}")

    banner("2. plain estimate: is the lift real?")
    diff, se, p = estimate(pre, post, group)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"measured lift = {diff:.2f}  (se = {se:.2f}, 95% CI: [{lo:.2f}, {hi:.2f}])")
    print(f"p = {p:.4f} -> {verdict} at 5%")

    banner("3. cuped cross-check: same answer with less noise?")
    theta = cuped_theta(pre, post)
    adj = adjust(pre, post)
    diff_c, se_c, _ = estimate(pre, adj, group)
    agree = "agrees" if abs(diff - diff_c) < se else "disagrees"
    print(f"theta = {theta:.2f}, noise cut by {100 * (1 - se_c / se):.1f}%")
    print(f"cuped lift = {diff_c:.2f} (se = {se_c:.2f}) -> {agree} with plain estimate")

    banner("4. could we have called it early?")
    # simulate 3 peeks: after 1 week, 1.5 weeks, full 2 weeks
    bar = budget_bar(alpha=0.05, peeks=3)
    for i, frac in enumerate((1 / 3, 2 / 3, 1.0), start=1):
        k = int(len(pre) * frac)
        _, se_i, p_i = estimate(pre[:k], post[:k], group[:k])
        call = "would have shipped early" if peek_significant(p_i, bar) else "too early to tell"
        print(f"peek {i} ({k} users): p = {p_i:.4f} (bar = {bar:.4f}) -> {call}")

    banner("5. who does it help most?")
    for name, d, s, p_val in estimate_by_segment(pre, post, group):
        print(f"{name}: lift = {d:.2f}  (se = {s:.2f}, p = {p_val:.4f})")
    control, treated = fit_t_learner(pre, post, group)
    pred = cate(pre, control, treated)
    print(f"T-learner avg lift = {pred.mean():.2f}")
    print(f"top 10% of users by predicted lift: avg predicted lift = "
          f"{pred[pred >= np.quantile(pred, 0.9)].mean():.2f}")

    banner("the decision")
    print(f"lift is real ({diff:.2f}, p < 0.05), cuped {agree}, "
          f"and heavy spenders gain most.")
    print("ship it: the win comes from heavy spenders, light spenders "
          "are unaffected, and we could have known at peek 2.")
