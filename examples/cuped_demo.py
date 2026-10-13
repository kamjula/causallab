"""cuped demo: run one experiment, print how much cuped shrinks the noise.

compares the plain difference-in-means against the cuped-adjusted
estimate on the same users, side by side."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuped import adjust, cuped_theta
from simulator import estimate, run_experiment


def report(name, diff, se):
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    print(f"{name}: lift = {diff:.2f}  (se = {se:.2f}, 95% CI: [{lo:.2f}, {hi:.2f}])")


if __name__ == "__main__":
    pre, post, group = run_experiment(n=2000, lift=5.0)

    # plain estimate: just compare post means
    diff, se, _ = estimate(pre, post, group)
    report("plain ", diff, se)

    # cuped: strip the pre-period part out of post, then estimate again
    theta = cuped_theta(pre, post)
    adj = adjust(pre, post)
    diff_c, se_c, _ = estimate(pre, adj, group)
    report("cuped ", diff_c, se_c)

    print()
    print(f"theta (how well pre predicts post) = {theta:.2f}")
    print(f"post variance: plain {post.var():.1f} -> cuped {adj.var():.1f}")
    print(f"noise (se) cut by {100 * (1 - se_c / se):.1f}%")
