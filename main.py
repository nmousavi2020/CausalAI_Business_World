
#
# Copyright (c) 2026 Naeim Mousavi
# All rights reserved.
#
# Description:
#     Causal AI and World Model workflow for structured business data,
#     combining causal analysis, scenario simulation, and neural prediction.
#
# Inputs:
#     Synthetic business data: price, marketing, discount
#
# Outputs:
#     data/business_data.csv
#     models/business_model.pt
#     plots/causal_effects.png
#     plots/scenario.png
#
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# BANNER
# =============================================================================

print()
print("╭─────────────────────────────────────────────────────────────╮")
print("│              CausalAI Business World — 2026                 │")
print("│       Causal AI + World Model + Structured Data ML          │")
print("│                                                             │")
print("│                Copyright © 2026 Naeim Mousavi               │")
print("╰─────────────────────────────────────────────────────────────╯")


# =============================================================================
# SETUP
# =============================================================================

np.random.seed(42)
torch.manual_seed(42)

for folder in ["data", "plots", "models"]:
    os.makedirs(folder, exist_ok=True)


# =============================================================================
# 1. BUSINESS DATA
# =============================================================================

n = 2000

df = pd.DataFrame({
    "price": np.random.uniform(10, 100, n),
    "marketing": np.random.uniform(0, 1000, n),
    "discount": np.random.uniform(0, 30, n)
})

df["sales"] = (
    500
    - 3 * df["price"]
    + 0.5 * df["marketing"]
    + 5 * df["discount"]
    + np.random.normal(0, 30, n)
)

df.to_csv("data/business_data.csv", index=False)

print("\n=== BUSINESS DATA ===")
print(f"Rows: {len(df)}")


# =============================================================================
# 2. CAUSAL AI
# =============================================================================

features = ["price", "marketing", "discount"]

X = df[features]
y = df["sales"]

causal_model = LinearRegression()
causal_model.fit(X, y)

print("\n=== CAUSAL EFFECTS ===")

for feature, effect in zip(features, causal_model.coef_):
    print(f"{feature:10s}: {effect:+.3f}")


# =============================================================================
# 3. CAUSAL EFFECT PLOT
# =============================================================================

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 5.5))

effects = causal_model.coef_

colors = [
    "#FF5C5C" if value < 0 else "#35D07F"
    for value in effects
]

bars = ax.barh(
    features,
    effects,
    color=colors,
    height=0.55
)

ax.axvline(
    0,
    color="white",
    linewidth=1,
    alpha=0.5
)

ax.set_title(
    "Causal AI — Business Drivers of Sales",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "Estimated Effect on Sales",
    fontsize=11
)

ax.set_xlim(left=-4)
ax.set_xlim(right=6)

for bar, value in zip(bars, effects):
    ax.text(
        value + (0.08 if value >= 0 else -0.08),
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.3f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=12,
        fontweight="bold"
    )

ax.grid(axis="x", alpha=0.15)

# White border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("white")
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    "plots/causal_effects.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="#111827"
)

plt.close()


# =============================================================================
# 4. WORLD MODEL
# =============================================================================

baseline_input = pd.DataFrame([{
    "price": df["price"].mean(),
    "marketing": df["marketing"].mean(),
    "discount": df["discount"].mean()
}])

scenario_input = pd.DataFrame([{
    "price": df["price"].mean() * 0.90,
    "marketing": df["marketing"].mean() * 1.10,
    "discount": df["discount"].mean()
}])

baseline = causal_model.predict(baseline_input)[0]
scenario = causal_model.predict(scenario_input)[0]

change = scenario - baseline

print("\n=== WORLD MODEL ===")
print(f"Baseline Sales : {baseline:.2f}")
print(f"Scenario Sales : {scenario:.2f}")
print(f"Sales Change   : {change:+.2f}")


# =============================================================================
# 5. WORLD MODEL PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

values = [baseline, scenario]
labels = ["Baseline", "Intervention"]

bars = ax.bar(
    labels,
    values,
    width=0.5,
    color=["#64748B", "#38BDF8"]
)

ax.set_title(
    "World Model — Business Scenario Simulation",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Predicted Sales")

ax.grid(
    axis="y",
    alpha=0.15
)

# White border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("white")
    spine.set_linewidth(1.2)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 8,
        f"{value:.1f}",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )

ax.text(
    0.5,
    max(values) * 0.55,
    f"Sales Impact: {change:+.1f}",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="#35D07F"
)

plt.tight_layout()

plt.savefig(
    "plots/scenario.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="#111827"
)

plt.close()


# =============================================================================
# 6. FOUNDATION MODEL
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
)

y_test = y_scaler.transform(
    y_test.values.reshape(-1, 1)
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003
)

loss_fn = nn.MSELoss()

for epoch in range(3000):

    prediction = model(X_train)

    loss = loss_fn(
        prediction,
        y_train
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================

with torch.no_grad():
    prediction = model(X_test).numpy()

prediction = y_scaler.inverse_transform(
    prediction
)

actual = y_scaler.inverse_transform(
    y_test.numpy()
)

mse = mean_squared_error(
    actual,
    prediction
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual,
    prediction
)

torch.save(
    model.state_dict(),
    "models/business_model.pt"
)

print("\n=== FOUNDATION MODEL ===")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print("Model saved.")

