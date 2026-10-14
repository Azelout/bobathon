# %% [markdown]
# # 02 — True OFF à enveloppe
#
# Estimateur unique `TrueOffEnvelopeRegressor` qui implémente les six blocs
# de la stratégie décrite dans `journal/STRATEGIE.md` (sections 4 et 11) :
#
# 1. Variables de visite et de traitement (sans cible)
# 2. Conversion de `on` en équivalent `off` (sans cible)
# 3. Corrections par évaluateur (apprises dans `fit`)
# 4. Variables par patient : droite pondérée + résumés + enveloppe 90 colonnes
# 5. HistGradientBoostingRegressor
# 6. Post-traitement par patient : parabole → isotonique → clip [0, 132]
#
# RMSE attendue : ≈ 2,90 (fourchette 2,82–2,95 par pli GroupKFold à 5 plis).
# L'estimateur et ses fonctions auxiliaires sont définis dans
# `src/parkinson/estimator.py` pour être correctement picklables.

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

# Joindre X et y pour le graphe DataOps
train = X_train.copy()
train["target"] = y_train.values

# %%
# Vérification rapide des colonnes disponibles
print("Colonnes X_train :", list(X_train.columns))
print("Lignes :", len(X_train), "| Patients :", X_train["patient_id"].nunique())

# %%
learner = build_learner(data_preview=train)

# %% [markdown]
# ## Évaluation croisée GroupKFold à 5 plis
#
# `skore.evaluate` détecte le `SkrubLearner` et utilise le `cv` et les
# `split_kwargs` déclarés au niveau du marqueur X.

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
project.put("02_true_off_envelope", report)

# %% [markdown]
# ## Rapport EstimatorReport sur le premier pli GroupKFold
#
# Le README exige une URL de rapport EstimatorReport.
# On utilise le premier pli d'un GroupKFold, jamais une séparation aléatoire.

# %%
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(
    gkf.split(X_train, y_train, groups=X_train["patient_id"])
)

X_tr_fold = train.iloc[train_idx].reset_index(drop=True)
X_val_fold = train.iloc[val_idx].reset_index(drop=True)

# Estimateur non fitté : skore le fitte sur X_train/y_train du pli
estimator_report = skore.EstimatorReport(
    TrueOffEnvelopeRegressor(),
    X_train=X_tr_fold.drop(columns=["target"]),
    y_train=X_tr_fold["target"],
    X_test=X_val_fold.drop(columns=["target"]),
    y_test=X_val_fold["target"],
)
print("RMSE premier pli :", estimator_report.metrics.rmse())
project.put("02_true_off_envelope_fold1_report", estimator_report)
print("EstimatorReport poussé : 02_true_off_envelope_fold1_report")

# %% [markdown]
# ## Entraînement sur tout le train et soumission

# %%
final_model = TrueOffEnvelopeRegressor()
final_model.fit(X_train, y_train)

X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
test_preds = final_model.predict(X_test)

submission = pd.DataFrame({"Index": X_test["Index"], "target": test_preds})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_true_off_envelope.csv", index=False)
print("Soumission écrite :", submission.shape)
print("Valeurs finies :", np.isfinite(test_preds).sum(), "/", len(test_preds))

# %%
