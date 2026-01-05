#File for baseline model: mean predictor

# ============================================================
# 0. IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "/Users/dariaproniakova/MLcourse/CS-433_Project_2/dataset/"

with open(DATA_PATH + "dataset.pkl", "rb") as f:
    data = pickle.load(f)

# ============================================================
# 2. BUILD INPUT FEATURES (STRAIN TENSOR)
# ============================================================

eps = pd.DataFrame(
    [d["eps_global"] for d in data],
    columns=["11", "22", "33", "23", "13", "12"]
)

# scale to [-1, 1]
strain_scaler = MinMaxScaler(feature_range=(-1, 1))
X = strain_scaler.fit_transform(eps.values)

# ============================================================
# 3. BUILD TARGET (FAILURE INDEX)
# ============================================================

plies = {}
angles = [0.0, 45.0, 90.0, -45.0]

for angle in angles:
    plies[angle] = pd.DataFrame([d["plies"][angle] for d in data])

stacked = pd.concat(plies, axis=1)
stacked.columns = [f"{a}_{m}" for a, m in stacked.columns]

# maximum failure index per sample
max_val = stacked.max(axis=1)
max_col = stacked.idxmax(axis=1)

values = stacked.to_numpy()
cols = stacked.columns.get_indexer(max_col)
y = values[np.arange(len(values)), cols]

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# optional standardization (does NOT affect baseline)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# 5. BASELINE: MEAN PREDICTOR
# ============================================================

# predict mean of training targets
y_mean = y_train.mean()
y_baseline_pred = np.full_like(y_test, fill_value=y_mean)

baseline_mse = mean_squared_error(y_test, y_baseline_pred)

print("===== BASELINE (MEAN PREDICTOR) =====")
print(f"Mean of y_train: {y_mean:.6f}")
print(f"Baseline MSE   : {baseline_mse:.6f}")

# ============================================================
# 6. OPTIONAL: VISUALIZATION
# ============================================================

plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_baseline_pred, s=8, alpha=0.4)
plt.axhline(y_mean, color="red", linestyle="--", label="Mean prediction")
plt.xlabel("True Failure Index")
plt.ylabel("Predicted Failure Index")
plt.title("Baseline Mean Predictor")
plt.legend()
plt.tight_layout()
plt.show()
