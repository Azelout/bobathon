"""Compare notre implémentation avec la référence scratch/pipeline/true_off.py."""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "pipeline"))
from true_off import TrueOffRegressor  # noqa: E402

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

gkf = GroupKFold(n_splits=5)
rmses_ref = []
for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups=X_train["patient_id"])):
    Xtr = X_train.iloc[tr].reset_index(drop=True)
    Xva = X_train.iloc[va].reset_index(drop=True)
    ytr = y_train.iloc[tr].reset_index(drop=True)
    yva = y_train.iloc[va].reset_index(drop=True)
    reg = TrueOffRegressor()
    reg.fit(Xtr, ytr)
    preds = reg.predict(Xva)
    rmse = float(np.sqrt(np.mean((preds - yva.values)**2)))
    rmses_ref.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

print(f"\nMean RMSE ref: {np.mean(rmses_ref):.4f} ± {np.std(rmses_ref):.4f}")
