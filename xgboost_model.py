# Package import

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost.callback import EarlyStopping
from itertools import product
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import data
with open("dataset.pkl", "rb") as f:
    data = pickle.load(f)

n = len(data)
print("n: "+str(n))
nbins = int(1 + 3.322 * np.log10(n))
# Define strain and plies FI values

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"],
)

print("Strain values")
print(eps.head().round(4))

# Scale strain values to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
eps_scaled = eps.copy()
eps_scaled.iloc[:, :] = scaler.fit_transform(eps.values)

print("Strain values scaled")
print(eps_scaled.head().round(2))
# Plies FI values
plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame(
        [d["plies"][angle] for d in data],
    )

print("Plies")
print(list(plies.keys()))
# Example ply FI values
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

# xgboost--------------------------------
X = eps_scaled.values  # shape: (n_samples, 6)
y = fail_summary["FI"].values  # regression target

N_SELECT = 20000

# Ensure we don’t try to select more samples than available
N = len(y)
assert N >= N_SELECT, f"Dataset too small: {N} samples available, {N_SELECT} requested."

# Random, reproducible selection of indices
rng = np.random.RandomState(42)
selected_idx = rng.choice(N, size=N_SELECT, replace=False)

# Subsample
X_small = X[selected_idx]
y_small = y[selected_idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

#these are the optimized hyperparameters:
n_estimators_list = [600]
max_depth_list = [5]
learning_rate_list =[0.175]

"""
#you can train the model using the following search range
n_estimators_list = [400,500,600]
max_depth_list = [2,3,4,5]
learning_rate_list = np.linspace(0.10, 0.25, 3)
"""

mse_list=[]
danger_ratio_list=[]
q_list=np.linspace(0.95,0.99,6)

for q in q_list:
    best_mse = float("inf")
    best_params = None
    best_danger_ratio = None
    for n_est, depth, lr in product(n_estimators_list, max_depth_list, learning_rate_list):

        xgb_reg = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            random_state=42
        )

        xgb_reg.set_params(
            objective='reg:quantileerror',
            quantile_alpha=q
        )

        xgb_reg.fit(
            X_train, y_train
        )

        y_pred = xgb_reg.predict(X_test)
        n_danger = np.sum(y_pred < y_test)
        danger_ratio=n_danger / len(y_pred)
        mse = mean_squared_error(y_test, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_params = (n_est, depth, lr)
            best_danger_ratio = danger_ratio

    mse_list.append(best_mse)
    danger_ratio_list.append(best_danger_ratio)
    print("Best MSE:", best_mse)
    print("Best parameters:")
    print("  n_estimators:", best_params[0])
    print("  max_depth:", best_params[1])
    print("  learning_rate:", best_params[2])
    print("  danger_ratio:", best_danger_ratio)
    print("  q: ", q)

# Create figure with 3/4 width-to-height ratio
fig, ax1 = plt.subplots(figsize=(10, 6))  # width/height = 3/4

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

#storing the predicted values:

#these are the best hyperparameters found earlier:
n_est = 600
depth= 5
lr =0.175

xgb_reg = xgb.XGBRegressor(
    n_estimators=n_est,
    max_depth=depth,
    learning_rate=lr,
    random_state=42
)

xgb_reg.set_params(
    objective='reg:quantileerror',
    quantile_alpha=0.99
)

xgb_reg.fit(
    X_train, y_train
)

# we chose to create the final prediction file only for last 15% of data (for faster computation)
n = X.shape[0]
split_idx = int(0.85 * n)
X_val = X[split_idx:]

y_pred = xgb_reg.predict(X_val)
df_pred = pd.DataFrame({"y_pred": y_pred})
# Save to CSV
df_pred.to_csv("results/xgboost_output.csv", index=False)
