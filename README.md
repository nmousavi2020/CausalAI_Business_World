# CausalAI Business World

**Causal AI + World Model + Structured Data ML**

A compact research-oriented workflow for structured business data, combining causal-style analysis, business scenario simulation, and neural-network prediction.

## Overview

The project studies how business variables influence sales and simulates business interventions such as:

* Lowering price
* Increasing marketing
* Predicting resulting sales

## Pipeline

```text
Structured Business Data
          ↓
     Causal Analysis
          ↓
      World Model
          ↓
   Neural Prediction
          ↓
 Business Insights
```

## Features

* Causal-style estimation of business effects
* World-model scenario simulation
* Neural network for sales prediction
* Standardized feature/target scaling
* RMSE and R² evaluation
* High-resolution business visualizations
* Reproducible synthetic dataset

## Results

Example results:

```text
Price Effect      : -3.032
Marketing Effect  : +0.503
Discount Effect   : +5.039

Baseline Sales    : 656.97
Scenario Sales    : 698.59
Sales Change      : +41.62

RMSE              : 32.27
R²                : 0.9694
```

## Project Structure

```text
CausalAI_Business_World/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── business_data.csv
│
├── models/
│   └── business_model.pt
│
└── plots/
    ├── causal_effects.png
    └── scenario.png
```

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete workflow:

```bash
python main.py
```

## Outputs

```text
data/business_data.csv
models/business_model.pt
plots/causal_effects.png
plots/scenario.png
```

## Note

The causal component uses multivariate linear regression to estimate causal-style effects on synthetic controlled data. It is intended as a compact research/portfolio demonstration rather than a full causal identification framework.

## Author

**Naeim Mousavi — 2026**

Copyright © 2026 Naeim Mousavi. All rights reserved.
