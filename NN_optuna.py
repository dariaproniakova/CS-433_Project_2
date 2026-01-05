# -*- coding: utf-8 -*-
"""
Code relative to the NN defintions, base model creation, and Optuna optimization to find best hyperparameters 
"""

# ============================================================
# 0. PACKAGE IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import csv
import pickle
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA & PREPARE FAILURE SUMMARY
# ============================================================
#IMPORTANT : Part 1 was provided by project supervisor to handle the given data

#Path to data file, change to given location
DATA_PATH = "/Users/dariaproniakova/MLcourse/CS-433_Project_2/dataset/"

with open(DATA_PATH + "dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data)
nbins = int(1 + 3.322 * np.log10(n))

# Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))

# Scale strain to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))
# Ply failure indices
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

max_val = stacked.max(axis=1)
max_col = stacked.idxmax(axis=1)
split = max_col.str.split("_", expand=True)
max_angle = split[0].astype(float)
short_mode = split[2]

values = stacked.to_numpy()
cols = stacked.columns.get_indexer(max_col)
FI = values[np.arange(len(values)), cols]

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
# 2. Neural Network Base Model Setup
# ============================================================

#Define Data Structure and Prepare Data

# NN learns only from epsilon features without knowing failure index (no data leakage, more robust!)
X = eps_scaled                                          # shape (N, 6)
# Store the maximum failure index for each sample
y = fail_summary["FI"].values.reshape(-1)               # shape (N, )

# First split: 85% train+val, 15% test
X_nn_temp, X_nn_test, y_nn_temp, y_nn_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Second split: from the 85%, make 70% train, 15% val (gives 70-15-15  ratio)
X_nn_train, X_nn_val, y_nn_train, y_nn_val = train_test_split(
    X_nn_temp, y_nn_temp, test_size=0.176, random_state=42
)

# Scale 
scaler = StandardScaler()
X_nn_train = scaler.fit_transform(X_nn_train)
X_nn_test = scaler.transform(X_nn_test)
X_nn_val = scaler.transform(X_nn_val)

# Convert to PyTorch tensors
X_nn_train = torch.tensor(X_nn_train, dtype=torch.float32)
X_nn_test = torch.tensor(X_nn_test, dtype=torch.float32)
X_nn_val = torch.tensor(X_nn_val, dtype=torch.float32) 

y_nn_train = torch.tensor(y_nn_train, dtype=torch.float32).view(-1,1)
y_nn_test = torch.tensor(y_nn_test, dtype=torch.float32).view(-1,1)
y_nn_val = torch.tensor(y_nn_val, dtype=torch.float32).view(-1,1) 

#Create Data Loaders

train_data = TensorDataset(X_nn_train, y_nn_train)
test_data = TensorDataset(X_nn_test, y_nn_test)
val_data = TensorDataset(X_nn_val, y_nn_val) 

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


#Define and Create NN model

#NN class definition
class FailureNet(nn.Module):
    def __init__(self):
        super(FailureNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Output: max failure index
        )

    def forward(self, x):
        return self.net(x)

#Create NN model
model = FailureNet()


#Loss function + optimizer

criterion = nn.MSELoss()  # same as regression models
optimizer = optim.Adam(model.parameters(), lr=0.001) #lr : learning rate (optimized later)


#NN Training Loop

num_epochs = 30

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_nn_test)
        val_loss = criterion(val_predictions, y_nn_test).item()
        test_losses.append(val_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Train MSE = {train_loss:.4f}, Test MSE = {val_loss:.4f}")


#Plot train and test losses


plt.plot(train_losses, label="Train error")
plt.plot(test_losses, label="Test error")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Neural Network Learning Curve")
plt.legend()
plt.show()

# ============================================================
# 3. NN optimization using Optuna
# ============================================================

#Necessary functions
def build_model(input_dim, n_layers, n_units, dropout):
    """
    Builds a fully-connected neural network for regression of max failure index.
    """

    layers = []

    # Input layer
    layers.append(nn.Linear(input_dim, n_units))
    layers.append(nn.ReLU())

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Hidden layers
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(n_units, n_units))
        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(n_units, 1))

    return nn.Sequential(*layers)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one full training epoch.
    Returns average loss over the dataset.
    """

    model.train()
    running_loss = 0.0

    for X_batch, y_batch in dataloader:

        # Move to CPU/GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)

        # Loss
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item() * X_batch.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test data.
    Returns average loss.
    """

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.detach().item() * X_batch.size(0)

    return running_loss / len(dataloader.dataset)


def objective(trial):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Hyperparameters
    #Uses Bayesian optimization with Optuna. Allows for efficient exploration
    #of hyperpapameter space through suggestions
    input_dim = X_nn_train.shape[1]
    n_layers = trial.suggest_int("n_layers", 2, 4)
    n_units = trial.suggest_int("n_units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

    # 2. Model
    model = build_model(input_dim, n_layers, n_units, dropout).to(device)

    # 3. Optimizer + loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()

    # 4. Training
    epochs = 40
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

    # 5. Validation
    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    return val_loss

#pruning to keep only promising results
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)

# ============================================================
# 4. Final model training + evaluation (BEST NN)
# ============================================================

print("\nTraining final NN with best Optuna hyperparameters")

best_params = study.best_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild best model
best_model = build_model(
    input_dim=X_nn_train.shape[1],
    n_layers=best_params["n_layers"],
    n_units=best_params["n_units"],
    dropout=best_params["dropout"],
).to(device)

optimizer = optim.Adam(
    best_model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
)
criterion = nn.MSELoss()

# Track losses
EPOCHS_FINAL = 30
train_mse = []
val_mse = []

for epoch in range(EPOCHS_FINAL):
    #Train 
    best_model.train()
    train_loss = train_one_epoch(
        best_model, train_loader, optimizer, criterion, device
    )
    train_mse.append(train_loss)

    # Validation 
    val_loss = evaluate(
        best_model, val_loader, criterion, device
    )
    val_mse.append(val_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Train={train_loss:.6f} | Val={val_loss:.6f}")

# ============================================================
# 5. Test set evaluation
# ============================================================

best_model.eval()
with torch.no_grad():
    y_test_pred = best_model(X_nn_test.to(device)).cpu().numpy().flatten()

y_test_np = y_nn_test.numpy().flatten()
residuals = y_test_np - y_test_pred

test_mse = np.mean((y_test_np - y_test_pred) ** 2)
print(f"\nFINAL TEST MSE (Best NN) = {test_mse:.6f}")

# ============================================================
# 6. Saving predictions
# ============================================================

results_df = pd.DataFrame({
    "y_true": y_test_np,
    "y_pred": y_test_pred,
    "residual": residuals,
})

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/nn_predictions.csv", index=False)

print("Predictions saved to results/nn_predictions.csv")

# ============================================================
# 7. Diagnostics plots
# ============================================================

# --- Learning curves ---
plt.figure(figsize=(7, 4))
plt.plot(train_mse, label="Train MSE")
plt.plot(val_mse, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("NN Training Curve (Best Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  Predicted vs True indexes
plt.figure(figsize=(5, 5))
plt.scatter(y_test_np, y_test_pred, s=10, alpha=0.4)
plt.plot(
    [y_test_np.min(), y_test_np.max()],
    [y_test_np.min(), y_test_np.max()],
    "k--",
)
plt.xlabel("True Failure Index")
plt.ylabel("Predicted Failure Index")
plt.title("NN: Predicted vs True (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual distribution
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Residual (True − Predicted)")
plt.ylabel("Count")
plt.title("NN Residual Distribution (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()
