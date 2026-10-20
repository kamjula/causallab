"""Heterogeneous treatment effects with a T-learner.

Segments split the data one way and estimate each slice; a T-learner
goes finer: it estimates the effect per user. Fit one plain linear
model on the control group and one on the treated group (both on the
pre-period covariate), then for every user predict both outcomes and
take the difference. That difference is the user's predicted lift;
the average of it is the average treatment effect."""

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_t_learner(pre, post, group):
    # one simple model per group, both using the pre-period covariate
    X = pre.reshape(-1, 1)
    control = LinearRegression().fit(X[group == 0], post[group == 0])
    treated = LinearRegression().fit(X[group == 1], post[group == 1])
    return control, treated


def cate(pre, control, treated):
    # predicted lift per user: treated prediction minus control prediction
    X = pre.reshape(-1, 1)
    return treated.predict(X) - control.predict(X)


def ate(pre, post, group):
    # average treatment effect = average of the per-user predicted lifts
    control, treated = fit_t_learner(pre, post, group)
    return cate(pre, control, treated).mean()
