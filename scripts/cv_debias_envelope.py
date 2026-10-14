import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson.estimator import DebiasEnvelopeRegressor

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]

gkf = GroupKFold(5)
folds = list(gkf.split(X, y, groups=X["patient_id"]))

rmses = []
for fold_i, (tr_idx, val_idx) in enumerate(folds):
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    m = DebiasEnvelopeRegressor()
    m.fit(X_tr, y_tr)
    p = m.predict(X_val)
    rmse = float(np.sqrt(np.mean((p - y_val.values) ** 2)))
    rmses.append(rmse)
    print(f"Fold {fold_i}: RMSE={rmse:.4f}")

print(f"\nMean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
