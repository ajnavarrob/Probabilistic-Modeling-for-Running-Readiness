# Probabilistic Modeling of Garmin Recovery Data

## Overview

This project fits a **Bayesian probit model** to Garmin-style running and recovery data in order to predict the probability of a **low-readiness day on the following day**.

The workflow assumes that the raw input files already exist and starts directly from those saved files. The script:

- loads the historical and future raw datasets
- engineers rolling recovery/training features
- defines the low-readiness target
- fits a Bayesian probit model using MAP estimation and a Laplace approximation
- generates posterior predictive probabilities
- evaluates model performance on historical and future data
- saves result tables and figures

---

## Required Input Files

Make sure the following files are present in the same directory as the modeling script:

- `garmin_history_raw.csv`
- `garmin_future_raw.csv`
- `garmin_model_metadata.json`

These files are treated as the starting point for the analysis.

---

## Main Script

Run the following script to reproduce the full probabilistic modeling pipeline:

```bash
python run_probabilistic_modeling.py
