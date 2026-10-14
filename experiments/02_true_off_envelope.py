# %% [markdown]
# # 02 — True OFF with envelope features
#
# A single `TrueOffEnvelopeRegressor` implementing the six blocks of the strategy
# described in `journal/STRATEGIE.md` (sections 4 and 11):
#
# 1. Visit and treatment features (target-free)
# 2. Convert `on` to an off-equivalent (target-free)
# 3. Rater-level scale corrections (learned in `fit`)
# 4. Per-patient weighted line + summaries + 90-column envelope
# 5. HistGradientBoostingRegressor
# 6. Per-patient post-processing: parabola → isotonic → clip [0, 132]
#
# Expected RMSE: ≈ 2.90 (range 2.82–2.95 per GroupKFold fold).
# The estimator and its helpers live in `src/parkinson/estimator.py`.

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.estimator import TrueOffEnvelopeRegressor, build_learner
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

# Merge X and y for the DataOps graph
train = X_train.copy()
train["target"] = y_train.values

# %%
print("X_train columns:", list(X_train.columns))
print("Rows:", len(X_train), "| Patients:", X_train["patient_id"].nunique())

# %%
learner = build_learner(data_preview=train)

# %% [markdown]
# ## Cross-validation — GroupKFold 5 folds
#
# `skore.evaluate` detects the `SkrubLearner` and uses the `cv` and
# `split_kwargs` declared on the X marker.

# %%
report = skore.evaluate(learner, data={"data": train})
print(report.metrics.rmse())

# %% [markdown]
# ## Push to Skore Hub

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_true_off_envelope", report)

# %% [markdown]
# ## EstimatorReport on fold 1
#
# The submission rules require a Skore Hub EstimatorReport URL.
# We use the first GroupKFold split — never a random holdout.

# %%
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
project.put("02_true_off_envelope_fold1_report", estimator_report)
print("EstimatorReport pushed: 02_true_off_envelope_fold1_report")

# %% [markdown]
# ## Train on the full training set and generate the submission CSV

# %%
final_model = TrueOffEnvelopeRegressor()
final_model.fit(X_train, y_train)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_true_off_envelope.csv", index=False)
print("Submission written:", submission.shape)
print("Finite predictions:", np.isfinite(test_preds).sum(), "/", len(test_preds))

# %%
