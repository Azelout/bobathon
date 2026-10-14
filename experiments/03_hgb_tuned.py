# %% [markdown]
# # 03 — HGB accordé + ensemble multi-graines
#
# Même architecture que 02 (blocs 1 à 6), avec :
# - `max_iter=1000`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=0.5`
# - Moyenne des prédictions sur 3 graines (0, 1, 2) via `SeedEnsembleRegressor`
#
# RMSE visée : < 2,90 (fourchette attendue 2,85–2,90 par pli GroupKFold à 5 plis).

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.estimator import (
    SeedEnsembleRegressor,
    TrueOffEnvelopeRegressor,
    build_learner_03,
)
from parkinson.hub import load_skore_credentials

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

train = X_train.copy()
train["target"] = y_train.values

# %%
print("Colonnes X_train :", list(X_train.columns))
print("Lignes :", len(X_train), "| Patients :", X_train["patient_id"].nunique())

# %%
learner = build_learner_03(data_preview=train)

# %% [markdown]
# ## Évaluation croisée GroupKFold à 5 plis

# %%
report = skore.evaluate(learner, data={"data": train})
print(report.metrics.rmse())

# %% [markdown]
# ## Push sur le Hub

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("03_hgb_tuned", report)

# %% [markdown]
# ## Rapport EstimatorReport sur le premier pli GroupKFold

# %%
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(
    gkf.split(X_train, y_train, groups=X_train["patient_id"])
)

X_tr_fold = train.iloc[train_idx].reset_index(drop=True)
X_val_fold = train.iloc[val_idx].reset_index(drop=True)

hgb_params = dict(
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.5,
)
estimator_report = skore.EstimatorReport(
    SeedEnsembleRegressor(seeds=[0, 1, 2], base_params=hgb_params),
    X_train=X_tr_fold.drop(columns=["target"]),
    y_train=X_tr_fold["target"],
    X_test=X_val_fold.drop(columns=["target"]),
    y_test=X_val_fold["target"],
)
print("RMSE premier pli :", estimator_report.metrics.rmse())
project.put("03_hgb_tuned_fold1_report", estimator_report)
print("EstimatorReport poussé : 03_hgb_tuned_fold1_report")

# %% [markdown]
# ## Entraînement sur tout le train et soumission

# %%
final_model = SeedEnsembleRegressor(seeds=[0, 1, 2], base_params=hgb_params)
final_model.fit(X_train, y_train)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "03_hgb_tuned.csv", index=False)
print("Soumission écrite :", submission.shape)
print("Valeurs finies :", np.isfinite(test_preds).sum(), "/", len(test_preds))

# %%
