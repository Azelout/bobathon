"""
Étapes finales de l'expérience 02_true_off_envelope :
1. EstimatorReport sur le premier pli GroupKFold → push Hub
2. Entraînement sur tout le train → prédiction X_test → submission CSV
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parkinson import PROJECT_ROOT
from parkinson.estimator import TrueOffEnvelopeRegressor
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_df = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
y_train = y_train_df["target"]
train = pd.concat([X_train, y_train.rename("target")], axis=1)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# ---------------------------------------------------------------------------
# 1. EstimatorReport sur le premier pli GroupKFold
# ---------------------------------------------------------------------------
print("=== EstimatorReport sur le premier pli ===")
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(
    gkf.split(X_train, y_train, groups=X_train["patient_id"])
)

X_tr_fold = train.iloc[train_idx].reset_index(drop=True)
X_val_fold = train.iloc[val_idx].reset_index(drop=True)

# Estimateur non fitté — skore le fitte sur X_train/y_train du pli
estimator_report = skore.EstimatorReport(
    TrueOffEnvelopeRegressor(),
    X_train=X_tr_fold.drop(columns=["target"]),
    y_train=X_tr_fold["target"],
    X_test=X_val_fold.drop(columns=["target"]),
    y_test=X_val_fold["target"],
)
rmse_fold = estimator_report.metrics.rmse()
print("RMSE premier pli :", rmse_fold)

cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_true_off_envelope_fold1_report", estimator_report)
print("EstimatorReport poussé : 02_true_off_envelope_fold1_report")

# ---------------------------------------------------------------------------
# 2. Entraînement sur tout le train + soumission
# ---------------------------------------------------------------------------
print("\n=== Entraînement final sur tout le train ===")
final_model = TrueOffEnvelopeRegressor()
final_model.fit(X_train, y_train)

test_preds = final_model.predict(X_test)
print("Valeurs finies :", np.isfinite(test_preds).sum(), "/", len(test_preds))
print("Min / Max :", test_preds.min(), "/", test_preds.max())

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_true_off_envelope.csv", index=False)
print("Soumission écrite :", submission.shape)
print("Premières lignes :")
print(submission.head())
