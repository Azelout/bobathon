"""
Run skore.evaluate on the full skrub DataOps graph and print per-fold RMSE.

Pushes the resulting CrossValidationReport to Skore Hub under the key
``02_true_off_envelope``.

Usage:
    python scripts/run_02_evaluate.py
"""
import time

import pandas as pd
import skore

from parkinson import PROJECT_ROOT
from parkinson.estimator import build_learner
from parkinson.hub import load_skore_credentials

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
train = X_train.copy()
train["target"] = y_train.values

print(f"Dataset: {train.shape} | Patients: {X_train['patient_id'].nunique()}")

learner = build_learner(data_preview=train)

t0 = time.time()
report = skore.evaluate(learner, data={"data": train})
elapsed = time.time() - t0

print("\n=== RMSE per fold ===")
print(report.metrics.rmse())
print(f"\nTotal time: {elapsed:.1f}s")

cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_true_off_envelope", report)
print("Report pushed to Hub: 02_true_off_envelope")
