# CausalLab — Experimentation & Causal Inference Workbench

An experimentation & causal inference workbench. I'm building this to get hands-on with what data science interviews at top companies actually test: designing experiments, reducing variance, and measuring real impact.

## Planned Capabilities

- A/B test simulator on realistic data
- CUPED variance reduction (smaller samples, same confidence)
- Sequential testing (stop early when there's signal)
- Heterogeneous treatment effects - who benefits most
- Streamlit dashboard with a live demo

## Current Implementation

The repository includes an A/B test simulator and a CUPED adjustment utility. CUPED variance-reduction performance and statistical correctness should be validated with repeatable tests before publishing numerical improvement claims.

## Development Progress

- [x] Week 1: repo setup
- [x] Week 1: data generator + experiment simulator (`src/simulator.py`)
- [x] CUPED adjustment utility implemented (`src/cuped.py`)
- [ ] Sequential testing (planned)
- [ ] Causal ML + dashboard (planned)
- [ ] Hosted demo + polish (planned)

## Run

```bash
pip install -r requirements.txt
python src/simulator.py
```

The simulator makes 2000 fake users, splits them into control/treatment,
adds a true lift of 5.0 to the treatment group, and prints the measured
lift with its standard error. Example output:

```
run 1: measured lift = 6.95  (se = 1.19, true lift = 5.0)
```
