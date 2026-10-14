"""
Final steps for experiment 02_true_off_envelope:

1. Build an EstimatorReport on the first GroupKFold split and push it to Skore Hub.
2. Train on the full training set, predict X_test, and write the submission CSV.

Usage:
    python scripts/run_02_final.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.estimator import TrueOffEnvelopeRegressor
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
train = pd.concat([X_train, y_train.rename("target")], axis=1)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# ---------------------------------------------------------------------------
# 1. EstimatorReport on the first GroupKFold split
# ---------------------------------------------------------------------------
print("=== EstimatorReport on fold 1 ===")
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))

X_tr_fold = train.iloc[train_idx].reset_index(drop=True)
X_val_fold = train.iloc[val_idx].reset_index(drop=True)

# Pass the unfitted estimator — skore fits it on the fold's train split.
estimator_report = skore.EstimatorReport(
    TrueOffEnvelopeRegressor(),
    X_train=X_tr_fold.drop(columns=["target"]),
    y_train=X_tr_fold["target"],
    X_test=X_val_fold.drop(columns=["target"]),
    y_test=X_val_fold["target"],
)
print("RMSE fold 1:", estimator_report.metrics.rmse())

cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_true_off_envelope_fold1_report", estimator_report)
print("EstimatorReport pushed: 02_true_off_envelope_fold1_report")

# ---------------------------------------------------------------------------
# 2. Train on the full training set and write the submission CSV
# ---------------------------------------------------------------------------
print("\n=== Final training on the full training set ===")
final_model = TrueOffEnvelopeRegressor()
final_model.fit(X_train, y_train)

test_preds = final_model.predict(X_test)
print("Finite predictions:", np.isfinite(test_preds).sum(), "/", len(test_preds))
print("Min / Max:", test_preds.min(), "/", test_preds.max())

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_true_off_envelope.csv", index=False)
print("Submission written:", submission.shape)
print(submission.head())
