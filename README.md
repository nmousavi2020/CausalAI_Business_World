# CausalAI Business World

**Causal AI + World Model + Structured Data ML + PyTorch Neural Network**

A compact research-oriented machine learning workflow for structured business data, combining causal analysis, business scenario simulation, and a PyTorch neural network for sales prediction.

## Overview

The project demonstrates how structured business data can be used to:

* Estimate business-variable effects
* Simulate business interventions
* Predict sales with a neural network
* Evaluate model performance using RMSE and R²
* Generate publication-quality visualizations

## Machine Learning Techniques

* **Linear Regression** — causal-style effect estimation
* **PyTorch Neural Network (MLP)** — sales prediction
* **StandardScaler** — feature and target normalization
* **Train/Test Split** — model evaluation
* **RMSE / R²** — predictive performance evaluation
* **Scenario Simulation** — business world-model analysis

## Workflow

```text
Structured Business Data
          ↓
   Linear Regression
     Causal Analysis
          ↓
      World Model
 Scenario Simulation
          ↓
 PyTorch Neural Network
     Sales Prediction
          ↓
    Business Insights
```

## Example Results

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
├── LICENSE
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

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Outputs

The workflow generates:

```text
data/business_data.csv
models/business_model.pt
plots/causal_effects.png
plots/scenario.png
```

## Research Note

The causal component uses multivariate linear regression to estimate causal-style effects on controlled synthetic data. It is a compact research and portfolio demonstration, not a full causal identification framework.

The neural prediction component uses a PyTorch multilayer perceptron (MLP) trained on standardized structured business data.

## Author

**Naeim Mousavi — 2026**

MIT License.
