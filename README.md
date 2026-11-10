# CausalLab — Experimentation & Causal Inference Workbench

> **Status: active development**

CausalLab is a Python workbench for learning and validating practical experimentation methods: randomized A/B simulation, treatment-effect estimation, sample-size planning, and CUPED variance reduction.

The repository separates **implemented and tested components** from planned work so unfinished features are not presented as complete.

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

### Tests

The pytest suite checks:

- known treatment differences
- no-lift behavior
- sample-size calculations
- invalid and non-finite inputs
- roughly balanced randomization
- recovery of injected treatment lift
- CUPED slope correctness
- CUPED variance reduction on correlated synthetic data
- edge cases that previously could return misleading NaN/Inf results

## Run locally

```bash
git clone https://github.com/kamjula/causallab.git
cd causallab

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/simulator.py
python -m pytest tests/ -q
```

## Current scope

The current code is a **statistical experimentation foundation**, not a complete causal-ML platform.

Planned work:

- [ ] Sequential testing
- [ ] Heterogeneous treatment-effect modeling
- [ ] Streamlit dashboard / visual experiment report
- [ ] Reproducible benchmark examples and documentation

## Why this project matters

The goal is not to hide experimentation behind a dashboard. The code keeps the statistical assumptions and failure cases visible so results can be inspected and challenged.

## Stack

Python · NumPy · pytest

## Author

**Sravani Kamjula** · AI/ML Engineer  
[LinkedIn](https://www.linkedin.com/in/sravani-kamjula/) · [Portfolio](https://sravaniportfoli.netlify.app/)
