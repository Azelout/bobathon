import pandas as pd
import numpy as np
from parkinson.estimator import DebiasWLSRegressor
from sklearn.model_selection import GroupKFold
import time

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]

gkf = GroupKFold(5)
folds = list(gkf.split(X, y, groups=X["patient_id"]))
tr_idx, val_idx = folds[0]
X_tr = X.iloc[tr_idx].reset_index(drop=True)
y_tr = y.iloc[tr_idx].reset_index(drop=True)
X_val = X.iloc[val_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

t0 = time.time()
m = DebiasWLSRegressor(quantile=0.90)
m.fit(X_tr, y_tr)
t1 = time.time()
p = m.predict(X_val)
t2 = time.time()
rmse = float(np.sqrt(np.mean((p - y_val.values) ** 2)))
print(f"RMSE fold 0: {rmse:.4f}")
print(f"fit: {t1-t0:.1f}s  predict: {t2-t1:.1f}s")
print(f"preds: min={p.min():.1f}  max={p.max():.1f}  finite={np.all(np.isfinite(p))}")
