# ============================================================
# Neural Network Quantile Regression with Pinball Loss
#
# This script retrains a neural network using an asymmetric
# (pinball / quantile) loss in order to reduce unsafe
# underestimation of the failure index.
#
# The network architecture is FIXED and taken from the
# previously optimized baseline NN (Optuna + MSE).
# Here, Optuna is only used to optimize:
#   - the quantile level q
#   - learning rate
#   - weight decay
# ============================================================


# ============================================================
# 0. PACKAGE IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna

sns.set_style("whitegrid")

# ============================================================
# 1. LOAD DATA & PREPARE FAILURE SUMMARY
# ============================================================
#IMPORTANT : Part 1 was provided by project supervisor to handle the given data

with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data) # Number of samples (used only for reporting / plots)
nbins = int(1 + 3.322 * np.log10(n))

# Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))

# Scale strain to [-1, 1] for numerical stability
scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))

# Failure indices are stored per ply and per failure mode.
# We extract the MAXIMUM failure index over all plies.
plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame(
        [d["plies"][angle] for d in data],
    )

print("Plies")
print(list(plies.keys()))

print("Failure index values for ply 0")
print(list(plies.values())[0].head().round(2))

# Failure summary

stacked = pd.concat(plies, axis=1)
stacked.columns = [f"{angle}_{mode}" for angle, mode in stacked.columns]
# Maximum FI per sample
max_val = stacked.max(axis=1)
max_col = stacked.idxmax(axis=1)
split = max_col.str.split("_", expand=True)
max_angle = split[0].astype(float)
short_mode = split[2]

values = stacked.to_numpy()
cols = stacked.columns.get_indexer(max_col)
FI = values[np.arange(len(values)), cols]
# Build a compact failure summary (for diagnostics only)
fail_threshold = 1.0
ffp = np.where(max_val >= fail_threshold, max_angle, np.nan)
mode = np.where(max_val >= fail_threshold, short_mode, "nf")

fail_summary = pd.DataFrame(
    {
        "ffp": np.where(np.isnan(ffp), "none", ffp),
        "mode": mode,
        "FI": FI,
    }
)

print("Failure summary")
print(fail_summary.sample(10, random_state=21))

# ============================================================
# 2. DATASET SPLIT
# ============================================================
# Standard 70 / 15 / 15 split:
#   - Train: used for weight updates
#   - Validation: used by Optuna to select hyperparameters
#   - Test: used ONLY for final evaluation
X = eps_scaled
y = FI

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)
# Standardization AFTER MinMax scaling improves NN convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# Data loaders for mini-batch training
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. FIXED BASELINE ARCHITECTURE
# ============================================================
# Architecture parameters taken from the baseline NN
# optimized with Optuna using MSE loss.
#
# These are FIXED here to isolate the effect of quantile training.
BEST_LAYERS  = 3
BEST_UNITS   = 83
BEST_DROPOUT = 2.1e-4

def build_model(input_dim):
    """
    Fully-connected feed-forward neural network.
    Activation: ReLU
    Optimizer: Adam
    """
    layers = [nn.Linear(input_dim, BEST_UNITS), nn.ReLU()]
    if BEST_DROPOUT > 0:
        layers.append(nn.Dropout(BEST_DROPOUT))

    for _ in range(BEST_LAYERS - 1):
        layers += [nn.Linear(BEST_UNITS, BEST_UNITS), nn.ReLU()]
        if BEST_DROPOUT > 0:
            layers.append(nn.Dropout(BEST_DROPOUT))

    layers.append(nn.Linear(BEST_UNITS, 1))
    return nn.Sequential(*layers)

# ============================================================
# 4. PINBALL LOSS
# ============================================================
# Pinball loss penalizes underestimation more strongly
# for high quantiles q.
#
# For a prediction y_hat and target y:
#   L_q = max(q * (y - y_hat), (q - 1) * (y - y_hat))
#
# This explicitly discourages unsafe predictions
# (y_hat < y).
def pinball_loss(y_pred, y_true, q):
    e = y_true - y_pred
    return torch.mean(torch.max(q * e, (q - 1) * e))

# ============================================================
# 5. OPTUNA OBJECTIVE (OPTIMIZE q)
# ============================================================
# Optuna searches for:
#   - quantile level q
#   - learning rate
#   - weight decay
#
# Objective: minimize validation pinball loss.

def objective(trial):

    q = trial.suggest_float("q", 0.90, 0.99)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    model = build_model(X_train_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Training loop
    for _ in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = pinball_loss(model(xb), yb, q)
            loss.backward()
            optimizer.step()

    # Validation pinball loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device))
        val_loss = pinball_loss(val_preds, y_val_t.to(device), q).item()

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best quantile model:", study.best_params)

# ============================================================
# 6. FINAL TRAINING WITH BEST q
# ============================================================

best_q  = study.best_params["q"]
best_lr = study.best_params["lr"]
best_wd = study.best_params["weight_decay"]

model = build_model(X_train_t.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)

train_loss_hist = []
val_loss_hist = []

EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = pinball_loss(model(xb), yb, best_q)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = pinball_loss(model(X_train_t.to(device)), y_train_t.to(device), best_q).item()
        val_loss   = pinball_loss(model(X_val_t.to(device)),   y_val_t.to(device),   best_q).item()

    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)

# ============================================================
# 7. TEST + METRICS
# ============================================================

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy().flatten()

residuals = y_test - y_pred
# Danger ratio = fraction of unsafe underpredictions
danger_ratio = np.mean(y_pred < y_test)
test_mse = mean_squared_error(y_test, y_pred)

print(f"\nFINAL RESULTS (q={best_q:.3f})")
print(f"Test MSE      : {test_mse:.6f}")
print(f"Danger ratio  : {danger_ratio:.4f}")

# ============================================================
# 8. SAVE CSV
# ============================================================

os.makedirs("results", exist_ok=True)
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "residual": residuals
}).to_csv("results/nn_quantile_predictions.csv", index=False)

# ============================================================
# 9. PLOTS
# ============================================================

# Learning curves
plt.figure(figsize=(6,4))
plt.plot(train_loss_hist, label="Train pinball loss")
plt.plot(val_loss_hist, label="Val pinball loss")
plt.xlabel("Epoch")
plt.ylabel("Pinball loss")
plt.legend()
plt.title(f"Quantile NN training (q={best_q:.3f})")
plt.grid(True)
plt.show()

# Predicted vs true
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, s=10, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("True FI")
plt.ylabel("Predicted FI")
plt.title("Quantile NN: Predicted vs True")
plt.grid(True)
plt.show()

# Residuals
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50)
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual distribution (Quantile NN)")
plt.grid(True)
plt.show()


# ============================================================
# 10. MSE & DANGER RATIO VS QUANTILE (ANALYSIS PLOT)
# ============================================================

q_list = np.linspace(0.90, 0.99, 11)
mse_list = []
danger_ratio_list = []

for q in q_list:

    model_q = build_model(X_train_t.shape[1]).to(device)
    optimizer_q = optim.Adam(
        model_q.parameters(),
        lr=best_lr,
        weight_decay=best_wd
    )

    # Train
    for _ in range(20):
        model_q.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_q.zero_grad()
            loss = pinball_loss(model_q(xb), yb, q)
            loss.backward()
            optimizer_q.step()

    # Test
    model_q.eval()
    with torch.no_grad():
        preds = model_q(X_test_t.to(device)).cpu().numpy().flatten()

    mse_list.append(mean_squared_error(y_test, preds))
    danger_ratio_list.append(np.mean(preds < y_test))

# ============================================================
# 11. FINAL PLOT (PUBLICATION QUALITY)
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=200)

ax1.plot(q_list, mse_list, marker="o", label="Test MSE")
ax1.set_xlabel("Quantile q")
ax1.set_ylabel("Test MSE")
ax1.set_ylim(0.0, 0.035)

ax2 = ax1.twinx()
ax2.plot(q_list, danger_ratio_list, marker="s",
         color="orange", label="Danger Ratio")
ax2.set_ylabel("Danger Ratio")
ax2.set_ylim(0.0, 0.11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("NN: Test MSE and Danger Ratio vs Quantile")
plt.grid(True)
plt.tight_layout()
plt.show()

