"""Lance skore.evaluate sur le graphe complet et affiche la RMSE par pli."""
import time
import pandas as pd
import numpy as np
import skore
import skrub
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# Charger les définitions
src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
lines = [l for l in src.split("\n") if not l.startswith("# %%")]
clean = "\n".join(lines)
stop = clean.find("# Chargement des données")
globs = {}
exec(clean[:stop], globs)

TrueOffEnvelopeRegressor = globs["TrueOffEnvelopeRegressor"]
build_learner = globs["build_learner"]

# Données
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
train = X_train.copy()
train["target"] = y_train.values

print(f"Dataset: {train.shape} | Patients: {X_train['patient_id'].nunique()}")

learner = build_learner(data_preview=train)

t0 = time.time()
report = skore.evaluate(learner, data={"data": train})
elapsed = time.time() - t0

print(f"\n=== RMSE par pli ===")
rmse_df = report.metrics.rmse()
print(rmse_df)
print(f"\nTemps total : {elapsed:.1f}s")
print(f"\ntype(report): {type(report).__name__}")

# Sauvegarder le rapport sur le Hub
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put("02_true_off_envelope", report)
print("Rapport poussé sur le Hub : 02_true_off_envelope")
