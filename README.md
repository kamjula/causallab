# CausalLab — Experimentation & Causal Inference Workbench

[![CI](https://github.com/kamjula/causallab/actions/workflows/ci.yml/badge.svg)](https://github.com/kamjula/causallab/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

CausalLab is a Python workbench for practical experimentation methods: randomized A/B simulation, treatment-effect estimation, sample-size planning, CUPED variance reduction, safe early peeking, segment-level lift, and per-user treatment effects with a T-learner. A small Streamlit dashboard puts the main methods side by side.

Every method is checked against synthetic data where the true effect is known, so each estimate can be compared with the truth.

## Implemented

### A/B experiment simulator

- Generates synthetic users with pre-period behavior
- Randomizes users into control/treatment
- Injects a known treatment lift
- Estimates treatment minus control
- Reports standard error, approximate p-value, and confidence interval
- Calculates required sample size for a target lift

### CUPED variance reduction

- Estimates the pre/post covariance adjustment coefficient
- Removes predictable pre-period variation from the post metric
- Validates mismatched lengths, non-finite values, zero variance, and too-small samples

### Sequential testing (early peeking)

- Splits the 5% false-positive budget evenly across the planned peeks (Bonferroni), so each peek has to clear a stricter bar
- In `examples/sequential_demo.py` (200 simulated experiments with no real lift, 5 peeks each), the plain 5% bar typically flags 13–19% of experiments; the split bar stays at or below about 5%
- Deliberately simple and conservative; alpha-spending designs such as O'Brien-Fleming are not implemented

### Segment-level lift

- Splits users into equal-sized segments by pre-period spend
- Runs the same lift estimate (lift, standard error, p-value) inside each segment, so a win concentrated in one group is not hidden by the overall average

### Heterogeneous treatment effects (T-learner)

- Fits one linear model on the control group and one on the treated group, both on the pre-period covariate
- Predicted lift per user = treated prediction minus control prediction; the average of these is the treatment-effect estimate
- On synthetic data where the lift grows with spend, it recovers that pattern (`examples/hte_demo.py`)

### Streamlit dashboard

Three pages: run an experiment, CUPED vs plain estimate, and who the change helps most.

### End-to-end example

`examples/end_to_end.py` takes one product question (keep or roll back a new checkout page) through the full workflow: plain estimate, CUPED cross-check, early peeks, segment lift, and the T-learner.

### Tests

The pytest suite (run in CI on every push) checks:

- known treatment differences
- no-lift behavior
- sample-size calculations
- invalid and non-finite inputs
- roughly balanced randomization
- recovery of injected treatment lift
- CUPED slope correctness
- CUPED variance reduction on correlated synthetic data
- edge cases that previously could return misleading NaN/Inf results
- the split peeking bar keeping false alarms near 5% while the plain bar does not
- segment estimates landing within 3 standard errors of the true per-segment lift
- the T-learner recovering how the lift changes with spend, and finding no effect when there is none

## Run locally

```bash
git clone https://github.com/kamjula/causallab.git
cd causallab

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/simulator.py              # A/B simulation, CIs, sample-size planning
python examples/cuped_demo.py        # plain vs CUPED estimate on the same data
python examples/sequential_demo.py   # false alarms from peeking, with and without the split bar
python examples/subgroup_demo.py     # lift per spend segment
python examples/hte_demo.py          # T-learner per-user lift vs the true lift
python examples/end_to_end.py        # one product decision through the whole workflow
streamlit run app.py                 # dashboard

pip install pytest
python -m pytest tests/ -q
```

## Limitations

- All data is synthetic; results show that the methods recover known effects, not performance on real product data
- The sequential test is a simple Bonferroni split, which is conservative
- The T-learner uses linear models on a single pre-period covariate
- Segments are based on one variable (pre-period spend)

## Why this project matters

The goal is not to hide experimentation behind a dashboard. The code keeps the statistical assumptions and failure cases visible so results can be inspected and challenged.

## Stack

Python · NumPy · pandas · SciPy · scikit-learn · Streamlit · pytest

## Author

**Sravani Kamjula** · AI/ML Engineer  
[LinkedIn](https://www.linkedin.com/in/sravani-kamjula/) · [Portfolio](https://sravaniportfoli.netlify.app/)
