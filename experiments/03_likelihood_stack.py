# %% [markdown]
# # 03 — Likelihood stack
#
# Deux ajouts à `TrueOffEnvelopeRegressor` (base : expérience 02, RMSE = 2,897) :
#
# 1. `sens_neighbors=True` — sensibilité empruntée aux voisins en `age_at_diagnosis`.
#    Corrélation de 0,65 entre patients partageant le même âge au diagnostic.
# 2. `use_posterior=True` — `OneSidedPosterior` (modèle bayésien L×S×C) empilé hors pli
#    comme variables `post_mean` et `post_sd` du boosting.
#
# RMSE attendue : ≈ 2,82 (fourchette 2,73–2,87 par pli GroupKFold à 5 plis).

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.estimator import (
    TrueOffEnvelopeRegressor,
    build_learner,
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

# %% [markdown]
# ## Essai rapide sur 300 patients (bug-check ~2 min)

# %%
rng = np.random.default_rng(42)
all_pids = X_train["patient_id"].unique()
sample_pids = rng.choice(all_pids, size=min(300, len(all_pids)), replace=False)
mask_sample = X_train["patient_id"].isin(sample_pids)
X_s = X_train[mask_sample].reset_index(drop=True)
y_s = y_train[mask_sample].reset_index(drop=True)

# Entraîner / prédire sur 300 patients pour détecter les bugs
split_idx = int(len(sample_pids) * 0.7)
tr_pids = sample_pids[:split_idx]
val_pids = sample_pids[split_idx:]

X_tr_s = X_s[X_s["patient_id"].isin(tr_pids)].reset_index(drop=True)
y_tr_s = y_s[X_s["patient_id"].isin(tr_pids)].reset_index(drop=True)
X_val_s = X_s[X_s["patient_id"].isin(val_pids)].reset_index(drop=True)
y_val_s = y_s[X_s["patient_id"].isin(val_pids)].reset_index(drop=True)

model_test = TrueOffEnvelopeRegressor(
    sens_neighbors=True,
    use_posterior=True,
    max_iter=100,  # rapide pour le bug-check
)
model_test.fit(X_tr_s, y_tr_s)
preds_test = model_test.predict(X_val_s)
rmse_test = float(np.sqrt(np.mean((preds_test - y_val_s.values) ** 2)))
print(f"Bug-check RMSE sur 300 patients (100 iter) : {rmse_test:.3f}")
print(f"Prédictions finies : {np.isfinite(preds_test).sum()} / {len(preds_test)}")
print(f"Min/Max : {preds_test.min():.1f} / {preds_test.max():.1f}")

# %% [markdown]
# ## SkrubLearner avec les deux options

# %%
learner = build_learner(
    data_preview=train,
    sens_neighbors=True,
    use_posterior=True,
)

# %% [markdown]
# ## Évaluation croisée GroupKFold à 5 plis
#
# ~15 minutes attendues (OneSidedPosterior × 5 plis × 3 sous-plis).

# %%
report = skore.evaluate(learner, data={"data": train})
rmse_scores = report.metrics.rmse()
print(rmse_scores)

# Vérification du seuil d'arrêt
mean_rmse = (
    float(rmse_scores.mean()) if hasattr(rmse_scores, "mean") else float(rmse_scores)
)
print(f"RMSE moyenne : {mean_rmse:.3f}")

if mean_rmse > 2.88:
    print("⚠️  RMSE > 2,88 — arrêt avant soumission. Scores par pli ci-dessus.")
    raise SystemExit(f"RMSE trop élevée : {mean_rmse:.3f}")

# %% [markdown]
# ## Push du rapport skore sur le Hub

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("03_likelihood_stack", report)

# %% [markdown]
# ## EstimatorReport sur le premier pli GroupKFold

# %%
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))

X_tr_fold = train.iloc[train_idx].reset_index(drop=True)
X_val_fold = train.iloc[val_idx].reset_index(drop=True)

estimator_report = skore.EstimatorReport(
    TrueOffEnvelopeRegressor(sens_neighbors=True, use_posterior=True),
    X_train=X_tr_fold.drop(columns=["target"]),
    y_train=X_tr_fold["target"],
    X_test=X_val_fold.drop(columns=["target"]),
    y_test=X_val_fold["target"],
)
print("RMSE premier pli :", estimator_report.metrics.rmse())
project.put("03_likelihood_stack_fold1_report", estimator_report)
print("EstimatorReport poussé : 03_likelihood_stack_fold1_report")

# %% [markdown]
# ## Entraînement sur tout le train et soumission

# %%
final_model = TrueOffEnvelopeRegressor(sens_neighbors=True, use_posterior=True)
final_model.fit(X_train, y_train)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "03_likelihood_stack.csv", index=False)
print("Soumission écrite :", submission.shape)
print("Valeurs finies :", np.isfinite(test_preds).sum(), "/", len(test_preds))

# %%
