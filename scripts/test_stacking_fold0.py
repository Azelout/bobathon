import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson.estimator import (
    DebiasWLSRegressor,
    StackingEnvelopeRegressor,
    TrueOffEnvelopeRegressor,
)

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]

gkf = GroupKFold(5)
folds = list(gkf.split(X, y, groups=X["patient_id"]))
tr_idx, val_idx = folds[0]
X_tr = X.iloc[tr_idx].reset_index(drop=True)
y_tr = y.iloc[tr_idx].reset_index(drop=True)
X_val = X.iloc[val_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

# Baseline HGB accordé
hgb_p = dict(
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.5,
)
t0 = time.time()
hgb = TrueOffEnvelopeRegressor(**hgb_p)
hgb.fit(X_tr, y_tr)
p_hgb = hgb.predict(X_val)
rmse_hgb = float(np.sqrt(np.mean((p_hgb - y_val.values) ** 2)))
print(f"HGB seul fold 0: RMSE={rmse_hgb:.4f}  ({time.time() - t0:.0f}s)")

# WLS seul
t0 = time.time()
wls = DebiasWLSRegressor(quantile=0.90)
wls.fit(X_tr, y_tr)
p_wls = wls.predict(X_val)
rmse_wls = float(np.sqrt(np.mean((p_wls - y_val.values) ** 2)))
print(f"WLS seul fold 0: RMSE={rmse_wls:.4f}  ({time.time() - t0:.0f}s)")

# Alpha optimal OLS
A = np.column_stack([p_hgb, p_wls])
alpha = np.linalg.solve(A.T @ A + 10 * np.eye(2), A.T @ y_val.values)
p_combo = A @ alpha
rmse_combo = float(np.sqrt(np.mean((p_combo - y_val.values) ** 2)))
print(f"Combo (oracle alpha) fold 0: RMSE={rmse_combo:.4f}  alpha={alpha}")
print("Note: oracle uses val labels, OOF stacking will be slightly worse")

# Stacking avec CV interne
t0 = time.time()
stack = StackingEnvelopeRegressor(hgb_params=hgb_p)
stack.fit(X_tr, y_tr)
p_stack = stack.predict(X_val)
rmse_stack = float(np.sqrt(np.mean((p_stack - y_val.values) ** 2)))
print(
    f"Stacking fold 0: RMSE={rmse_stack:.4f}"
    f"  alpha={stack.alpha_}  ({time.time() - t0:.0f}s)"
)
