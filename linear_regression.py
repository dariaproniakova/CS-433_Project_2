# -*- coding: utf-8 -*-
"""MLproject2.ipynb"""

#Package import

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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor


with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data)
nbins = int(1 + 3.322 * np.log10(n))

#  Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))


scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))

plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame(
        [d["plies"][angle] for d in data],
    )

print("Plies")
print(list(plies.keys()))
#
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

# ------------------------------
# BUILD FEATURE MATRIX
# ------------------------------

# Concatenate epsilon and the failure indexes (contained in stacked) along 1 axis
X = pd.concat([eps_scaled, stacked], axis=1).values     # shape (N, 22)
# Store the maximum failure index for each sample
y = fail_summary["FI"].values.reshape(-1)               # shape (N, )

"""Polynomial feature implementation"""


# -------------------------------------------
# 1. Reduce dataset to 10k samples (to avoid memory problems)
# -------------------------------------------

# Desired subset size
N_SELECT = 10000

# Ensure we don’t try to select more samples than available
N = len(y)
assert N >= N_SELECT, f"Dataset too small: {N} samples available, {N_SELECT} requested."

# Random, reproducible selection of indices
rng = np.random.RandomState(42)
selected_idx = rng.choice(N, size=N_SELECT, replace=False)

# Subsample
X_small = X[selected_idx]
y_small = y[selected_idx]

# -------------------------------------------
# 2. Create new train-test split variables for polynomial features
# -------------------------------------------
poly_X_train, poly_X_test, poly_y_train, poly_y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

# -------------------------------------------
# 3. Build new pipeline for polynomial training
# -------------------------------------------

q_list=np.linspace(0.87,0.96,5)
danger_ratio_list=[]
mse_list=[]
legend_list=["without pinball loss", "with pinball loss"]
####
for indx,q in enumerate(q_list):

    poly_pipeline = Pipeline([
        ('poly_scaler', StandardScaler()),
        ('poly_expansion', PolynomialFeatures(include_bias=False)),
        ('quantile_model', QuantileRegressor(quantile=q))
    ])
    """
    #this section can be used for training:
    poly_param_grid = {
        'poly_expansion__degree': [1,2,3],
        'quantile_model__alpha': np.logspace(0.01,10,5),
    }
    """
    # optimized parameters are the following: 
    poly_param_grid = {
        'poly_expansion__degree': [3],
        'quantile_model__alpha': [1.023292992280754],
    }

    poly_grid_search = GridSearchCV(
        estimator=poly_pipeline,
        param_grid=poly_param_grid,
        scoring='neg_mean_absolute_error',   # still OK; model internally uses pinball loss
        cv=3,
        verbose=0
    )

    poly_grid_search.fit(poly_X_train, poly_y_train)

    print("\nBest polynomial model parameters:", poly_grid_search.best_params_)
    print("Best CV MSE:", -poly_grid_search.best_score_)

    best_poly_model = poly_grid_search.best_estimator_
    poly_y_pred = best_poly_model.predict(poly_X_test)

    print("\nPolynomial Model Test MSE:", mean_squared_error(poly_y_test, poly_y_pred))
    mse_list.append(mean_squared_error(poly_y_test, poly_y_pred))
    print("Polynomial Model Test R²:", r2_score(poly_y_test, poly_y_pred))

    n_danger = np.sum(poly_y_pred < poly_y_test)
    danger_ratio = n_danger / len(poly_y_pred)
    print(danger_ratio)
    danger_ratio_list.append(danger_ratio)

    """ #this part can be used to generate additional plot (prediction with and without pinball loss)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        poly_y_test,
        poly_y_pred,
        marker='.',
        alpha=0.5,
        label=legend_list[indx]
    )

    # Ideal y = x line
    min_val = min(poly_y_test.min(), poly_y_pred.min())
    max_val = max(poly_y_test.max(), poly_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=2, label='Ideal')

    plt.xlabel("True Failure Index", fontsize=20)
    plt.ylabel("Predicted Failure Index", fontsize=20)
    plt.title("Predicted vs True", fontsize=20)

    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=18)

    plt.tight_layout()
plt.show()"""

# plotting danger ratio and mse versus quantile
fig, ax1 = plt.subplots(figsize=(10, 6))

# First Y axis (left): MSE
ax1.plot(
    q_list,
    mse_list,
    label="MSE",
    marker='o',
    markersize=6,
    linewidth=2
)
ax1.set_xlabel("Quantile q", fontsize=20)
ax1.set_ylabel("MSE", fontsize=20)
ax1.tick_params(axis='both', labelsize=18)

# Second Y axis (right): danger ratio
ax2 = ax1.twinx()
ax2.plot(
    q_list,
    danger_ratio_list,
    color='orange',
    label="Danger Ratio",
    marker='s',
    markersize=6,
    linewidth=2
)
ax2.set_ylabel("Danger Ratio", fontsize=20)
ax2.tick_params(axis='y', labelsize=18)

# Combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines + lines2,
    labels + labels2,
    loc='upper left',
    fontsize=18
)

# Title
plt.title("MSE and Danger Ratio vs Quantile", fontsize=20)

plt.tight_layout()
plt.show()

#storing the predicted values----------------------
poly_pipeline = Pipeline([
        ('poly_scaler', StandardScaler()),
        ('poly_expansion', PolynomialFeatures(include_bias=False)),
        ('quantile_model', QuantileRegressor(quantile=0.96))
    ])
poly_param_grid = {
        'poly_expansion__degree': [3], #these are the best hyperparameters found earlier
        'quantile_model__alpha': [1.023292992280754],
    }
poly_grid_search = GridSearchCV(
    estimator=poly_pipeline,
    param_grid=poly_param_grid,
    scoring='neg_mean_absolute_error',   # still OK; model internally uses pinball loss
    cv=3,
    verbose=0
)

# we chose to create the final prediction file only for last 15% of data (for faster computation)
n = X.shape[0]
split_idx = int(0.85 * n)
X_val = X[split_idx:]

poly_grid_search.fit(poly_X_train, poly_y_train)
best_poly_model = poly_grid_search.best_estimator_
poly_y_pred = best_poly_model.predict(X_val) #the prediction is done for X_val
df_pred = pd.DataFrame({"poly_y_pred": poly_y_pred})
# Save to CSV
df_pred.to_csv("results/linear_regression_output.csv", index=False)
