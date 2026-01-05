[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)

Prediction of Carbon Fibre Composite Failure Project
Team: Daria Proniakova, Michael Kornbeck, Idil Sonmez
Hosting Lab: LPAC, COMATEC
Name of contact person: Boschert Pierre-Alexandre

This repository contains the code used for the CS-433 project on machine-learning-based surrogate modeling of composite failure indices. The goal is to predict the maximum failure index of a laminate from global strain data, with a strong emphasis on safety-critical behavior.

All code is written in Python and is fully reproducible.

Repository Structure:

├── dataset.pkl             # Full dataset provided by the hosting lab
├── demo.py                 # Original demo file provided by the lab for data exploration and experiments
├── linear_regression.py    #linear reg model
├── baseline.py             #baseline mean predictor
├── xgboost_model.py        #xgboost
├── NN_optuna.py            #Neural Network baseline, optimization with optuna to pretrain model for founding best N for layers and unis for further use for quantile NN
├── NN_quantile.py          #Neural Network for quntile loss - for safety reasons
└── README.md               #this file



Description of each file and its output if exist:

1) Dataset Description (dataset.pkl):
The file dataset.pkl contains a list of dictionaries, where each element corresponds to one simulated laminate configuration.

Structure of One Sample
Each element d in data has the following structure:
d = {
    "eps_global": array-like of length 6,
    "plies": {
        0.0:  {ft, fc, mt, mc},
        45.0: {ft, fc, mt, mc},
        90.0: {ft, fc, mt, mc},
        -45.0:{ft, fc, mt, mc}
    }
}

1. Global strain vector (d["eps_global"])
The global laminate strain is a 6D vector:
e = (eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12)
These values describe the macroscopic deformation applied to the laminate.

2. Ply-level failure indices (d["plies"][angle])
For each ply orientation p = (0,45,90,-45) #degrees
four failure modes are evaluated:
ft — fibre tension
fc — fibre compression
mt — matrix tension
mc — matrix compression
This results in 16 failure indices per sample.

3. Regression Target
Rather than predicting all 16 indices, all models focus on predicting a single scalar target: y_max = max F(p,c)
This maximum failure index corresponds to the earliest failure initiation and is the most relevant quantity for engineering decision-making.

FILE- demo.py 
The file `demo.py` was provided by the hosting laboratory together with the dataset.  
It was used solely for exploratory purposes, to understand the dataset structure and visualize strain and failure-index distributions. All modeling, preprocessing, and analysis code in this repository was implemented independently.


2) Data Preparation (Common for all models):
- Extract global strain features
- Scale strain components to [−1,1] using MinMaxScaler
- Compute the maximum failure index per sample
- Random subsampling (for computational feasibility)
- Train–test split with fixed random seed for reproducibility (80-20 for LR and XGBoost and 70-15-15 for NN for train and test)

3) Linear Regression (Polynomial + Quantile) - file: linear_regression.py
Model Description
- Polynomial regression implemented via PolynomialFeatures
- L2-regularized quantile regression using QuantileRegressor
- Degree limited to 3 due to computational constraints
Key Characteristics
- Dataset reduced to 10,000 samples
- Train–test split: 80–20
- Pinball (quantile) loss used to penalize unsafe underprediction
- Quantile sweep performed over: q=[0.87,0.96]
Outputs
- MSE vs quantile plot
- Danger ratio vs quantile plot
- CSV file containing final predictions: linear_regression_output.csv

4) XGBoost Regression (Quantile Objective)- file: xgboost_model.py
Model Description
- Gradient-boosted decision trees (XGBRegressor)
- Uses the native quantile regression objective: objective='reg:quantileerror'
- Designed to capture nonlinear interactions between strain components
Key Characteristics
- Dataset reduced to 20,000 samples
- Train–test split: 80–20
- Hyperparameters fixed after empirical optimization:
n_estimators = 600
max_depth = 5
learning_rate = 0.175
- Quantile sweep performed over: q=[0.95,0.99]
Outputs
- MSE vs quantile plot
- Danger ratio vs quantile plot
- CSV file containing final predictions: xgboost_output.csv

5) Neural Networks Models
Two neural network pipelines are implemented:
- Baseline neural network trained with MSE loss - file: NN_optuna.py
- Safety-aware neural network retrained using pinball (quantile) loss - file: NN_quintile.py
Both models share the same data preprocessing and architecture design principles.

Network architecture:
All neural networks are fully connected feed-forward models with the following properties:
Input dimension: 6 (global strain components)
Output dimension: 1 (maximum failure index)
Hidden layers: 2–4 layers (optimized via Optuna) - best is 3
Activation function: ReLU
Regularization: Dropout and L2 weight decay
Output layer: Linear (no activation)
ReLU activations are used throughout due to their numerical stability and efficiency when training deep regression models.

Optimizer and Training Setup:
All neural networks are trained using:
Optimizer: Adam
Loss function (baseline): Mean Squared Error (MSE)
Loss function (quantile models): Pinball (quantile) loss
Batch training: Mini-batch gradient descent
Batch size: 32–128 (optimized)
Adam is chosen due to its adaptive learning rate and robustness to noisy gradients, which is particularly important given the large dataset size.

Data Splitting and Scaling
The dataset is split as follows: 70% training - 15% validation - 15% test
All input features are standardized using StandardScaler after splitting, ensuring no data leakage.
Targets are not scaled to preserve their physical meaning.

Baseline Neural Network with Optuna Optimization - file: NN_optuna.py
Hyperparameter Optimization
Hyperparameters are optimized using Optuna, with the validation MSE as the objective.
The following parameters are tuned:
| Hyperparameter    | Search Range          |
| ----------------- | --------------------- |
| Number of layers  | 2 – 4                 |
| Neurons per layer | 32 – 128              |
| Dropout rate      | 0.0 – 0.4             |
| Learning rate     | (10^{-5}) – (10^{-2}) |
| Weight decay      | (10^{-6}) – (10^{-1}) |
| Batch size        | {32, 64, 128}         |


Each trial is trained for a fixed number of epochs, and poorly performing trials are pruned early using Optuna’s MedianPruner to reduce computational cost.

Outputs
Learning curves (train vs validation MSE)
Predicted vs true failure index scatter plot
Residual distribution
CSV file with test predictions: results/nn_predictions.csv


Safety-Aware Neural Network (Quantile Training) - file: NN_quantile.py
To address the ethical risk of unsafe underestimation, the neural network is retrained using pinball (quantile) loss, which penalizes underprediction more strongly than overprediction.
Pinball loss: 
For a chosen quantile q, the pinball loss is defined as:

e = y_true - y_pred
    return torch.mean(torch.max(q * e, (q - 1) * e))

For q>0.5, underestimations receive higher penalties, encouraging conservative predictions.

Quantile Optimization Strategy
-- Architecture fixed to the best configuration found in the baseline Optuna study
-- Optuna used again, but only to optimize training hyperparameters:
    - Learning rate
    - Weight decay
-- Objective: Validation pinball loss
Evaluation Metrics

For each quantile q, the following metrics are computed:
Test MSE
Test pinball loss
Danger ratio - danger_ratio = np.mean(y_pred < y_true)

Outputs
- MSE vs quantile plot
- Danger ratio vs quantile plot
- Pinball loss vs quantile plot
- CSV file containing final predictions

6) Baseline Mean Predictor - file baseline.py
This script implements a mean-value predictor that always outputs the average training failure index. It serves as a lower-bound baseline to verify that all learned models provide meaningful performance improvements.

The script reports:
- the mean failure index computed on the training set,
- the baseline test MSE,
- an optional visualization comparing constant predictions to true failure indices.

This baseline is intentionally simplistic. Its role is not to achieve good accuracy, but to demonstrate that linear regression, XGBoost, and neural network models significantly outperform a naive predictor and therefore extract meaningful information from the strain data.


Purpose

REPRODUCIBILITY:
To reproduce results:
1. Place dataset.pkl in the working directory
2. Install dependencies:
    pip install -r requirements.txt   #numpy, pandas, scikit-learn, scikit-learn, seaborn, xgboost, torch, optuna
3. Run model scripts
    Linear regression (polynomial + quantile loss): 
        python linear_regression.py
    XGBoost with quantile objective:
        python xgboost_model.py
    Neural network baseline + Optuna optimization:
        python NN_optuna.py
    Neural network retraining with pinball (quantile) loss:
        python nn_quantile_optuna.py

NOTE: 
### Dataset and Git LFS
The dataset (`dataset.pkl`, ~456 MB) is tracked using Git Large File Storage (Git LFS).
After cloning the repository, ensure Git LFS is installed and run:

```bash
git lfs install
git lfs pull

