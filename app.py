"""CausalLab dashboard. Runs the experiment simulator with sliders
and shows the measured lift live. CUPED page compares plain vs adjusted
estimates; the HTE page shows who gains the most per spend segment."""

import streamlit as st
import numpy as np
import pandas as pd

from src.simulator import run_experiment, estimate, sample_size
from src.cuped import adjust
from src.hte import fit_t_learner, cate


def experiment_page():
    st.header("run an experiment")

    with st.sidebar:
        st.subheader("settings")
        n = st.slider("users", 500, 20000, 2000, step=500)
        lift = st.slider("true lift", 0.0, 20.0, 5.0, step=0.5)
        go = st.button("run it")

    if not go:
        st.write("pick the users and the lift, then hit run.")
        return

    pre, post, group = run_experiment(n=n, lift=lift)
    diff, se, p = estimate(pre, post, group)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    verdict = "significant" if p < 0.05 else "not significant"

    c1, c2, c3 = st.columns(3)
    c1.metric("measured lift", f"{diff:.2f}")
    c2.metric("p-value", f"{p:.4f}")
    c3.metric("verdict (5%)", verdict)

    st.write(f"95% CI: [{lo:.2f}, {hi:.2f}]")
    st.write(f"noise check: this needed {sample_size(lift or 1.0, post.std())} "
             "users per group to spot the lift")

    means = pd.DataFrame({
        "group": ["control", "treatment"],
        "average spend": [post[group == 0].mean(), post[group == 1].mean()],
    }).set_index("group")
    st.bar_chart(means)


def cuped_page():
    st.header("cuped vs plain")

    with st.sidebar:
        st.subheader("settings")
        n = st.slider("users", 500, 20000, 2000, step=500)
        lift = st.slider("true lift", 0.0, 20.0, 5.0, step=0.5)
        go = st.button("run it")

    if not go:
        st.write("same experiment measured two ways: plain means vs "
                 "cuped-adjusted. hit run to see which one is noisier.")
        return

    pre, post, group = run_experiment(n=n, lift=lift)

    d0, se0, p0 = estimate(pre, post, group)
    adj = adjust(pre, post)
    d1, se1, p1 = estimate(pre, adj, group)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("plain")
        st.metric("measured lift", f"{d0:.2f}")
        st.metric("standard error", f"{se0:.2f}")
        st.metric("p-value", f"{p0:.4f}")
    with c2:
        st.subheader("cuped")
        st.metric("measured lift", f"{d1:.2f}")
        st.metric("standard error", f"{se1:.2f}")
        st.metric("p-value", f"{p1:.4f}")

    # how much of the noise cuped removed, and what it buys in sample size
    smaller = 1 - se1 ** 2 / se0 ** 2
    st.write(f"cuped cut the noise variance by {smaller:.0%}. "
             "the measured lift stays about the same, it is just "
             "measured with a steadier hand.")
    st.write(f"users per group to spot the lift: "
             f"{sample_size(lift or 1.0, post.std())} plain vs "
             f"{sample_size(lift or 1.0, adj.std())} with cuped")


def hte_page():
    st.header("who does it help most")

    with st.sidebar:
        st.subheader("settings")
        n = st.slider("users", 500, 20000, 4000, step=500)
        go = st.button("run it")

    if not go:
        st.write("the change helps heavy spenders more than light ones. "
                 "hit run and the T-learner figures out who gains the most.")
        return

    # the true effect grows with pre-period spend, like the hte demo
    pre, post, group = run_experiment(n=n, lift=0.0)
    true_lift = 2.0 + 0.06 * (pre - 100.0)
    post[group == 1] += true_lift[group == 1]

    control, treated = fit_t_learner(pre, post, group)
    pred = cate(pre, control, treated)

    diff, se, p = estimate(pre, post, group)
    st.write(f"plain measured lift: {diff:.2f} (se = {se:.2f}, p = {p:.4f}). "
             f"T-learner's average lift: {pred.mean():.2f}.")

    # average predicted lift per spend segment
    edges = np.quantile(pre, [0, 0.25, 0.5, 0.75, 1.0])
    seg = np.clip(np.digitize(pre, edges[1:-1]), 0, 3)
    names = ["lightest", "light", "heavy", "heaviest"]
    means = pd.DataFrame({
        "segment": names,
        "predicted lift": [pred[seg == i].mean() for i in range(4)],
    }).set_index("segment")
    st.bar_chart(means)
    st.write("the bar grows left to right: the more a user spent before, "
             "the more the change helps them.")


def main():
    st.title("CausalLab")
    page = st.sidebar.radio(
        "page", ["run experiment", "cuped vs plain", "who does it help most"])
    if page == "run experiment":
        experiment_page()
    elif page == "cuped vs plain":
        cuped_page()
    else:
        hte_page()


if __name__ == "__main__":
    main()
