"""CausalLab dashboard. Runs the experiment simulator with sliders
and shows the measured lift live. More pages (CUPED, HTE) come next."""

import streamlit as st
import pandas as pd

from src.simulator import run_experiment, estimate, sample_size


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


def main():
    st.title("CausalLab")
    page = st.sidebar.radio("page", ["run experiment"])
    if page == "run experiment":
        experiment_page()


if __name__ == "__main__":
    main()
