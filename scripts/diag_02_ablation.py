"""
Diagnostic : compare RMSE avec vs sans enveloppe sur 1 pli du dataset complet.
Aide à détecter si le gain de l'enveloppe est le bon.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT

src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
lines = [l for l in src.split("\n") if not l.startswith("# %%")]
clean = "\n".join(lines)
stop = clean.find("# Chargement des données")
globs = {}
exec(clean[:stop], globs)

TrueOff = globs["TrueOffEnvelopeRegressor"]
HistGB = globs["HistGradientBoostingRegressor"]
_block1 = globs["_block1_visit_features"]
_apply_on_eq = globs["_apply_on_eq"]
_apply_corrections = globs["_apply_corrections"]
_fit_patient_line = globs["_fit_patient_line"]
_patient_summary = globs["_patient_summary_features"]
_envelope = globs["_envelope_features"]
_postprocess = globs["_postprocess_predictions"]
RAW_NUM = globs["RAW_NUM_COLS"]

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
print("Dataset:", X_train.shape)

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
Xtr = X_train.iloc[tr].reset_index(drop=True)
Xva = X_train.iloc[va].reset_index(drop=True)
ytr = y_train.iloc[tr].reset_index(drop=True)
yva = y_train.iloc[va].reset_index(drop=True)
print(f"Train: {Xtr.shape}, Val: {Xva.shape}")

# Entraîner le modèle complet
t0 = time.time()
model = TrueOff()
model.fit(Xtr, ytr)
print(f"fit: {time.time()-t0:.1f}s")

preds = model.predict(Xva)
rmse_full = float(np.sqrt(np.mean((preds - yva.values)**2)))
print(f"\nRMSE complet (avec enveloppe): {rmse_full:.4f}")

# RMSE sans post-traitement
sidx = model._sorted_idx(Xva)
Xva_s = Xva.iloc[sidx].reset_index(drop=True)
df = _block1(Xva_s)
df = _apply_on_eq(df, model.drug_curve_, model.patient_sensitivity_, model.nboth_)
df = _apply_corrections(df, model.rater_lines_, model.on_effect_, model.on_offset_, model.rater_q95_)
df = _fit_patient_line(df)
df = _patient_summary(df)
df_env = _envelope(df)
feat = df_env[model.feature_cols_]
raw_preds = model.booster_.predict(feat.values)
result_raw = np.empty(len(Xva))
result_raw[sidx] = raw_preds
rmse_raw = float(np.sqrt(np.mean((result_raw - yva.values)**2)))
print(f"RMSE sans post-traitement: {rmse_raw:.4f}")

# RMSE sans enveloppe (réentraîner sans les colonnes env_*)
env_cols = [c for c in model.feature_cols_ if c.startswith("env_")]
no_env_cols = [c for c in model.feature_cols_ if not c.startswith("env_")]
print(f"\nNombre de colonnes env_: {len(env_cols)}")
print(f"Nombre de colonnes sans env_: {len(no_env_cols)}")

# Réentraîner sans les colonnes d'enveloppe
df_tr = _block1(Xtr)
df_tr = _apply_on_eq(df_tr, model.drug_curve_, model.patient_sensitivity_, model.nboth_)
df_tr = _apply_corrections(df_tr, model.rater_lines_, model.on_effect_, model.on_offset_, model.rater_q95_)
df_tr = _fit_patient_line(df_tr)
df_tr = _patient_summary(df_tr)
df_tr_env = _envelope(df_tr)

# Aligner y sur le tri
sidx_tr = model._sorted_idx(Xtr)
ytr_s = pd.Series(ytr.to_numpy()[sidx_tr])

feat_no_env_tr = df_tr_env[no_env_cols]
hgb_no_env = HistGB(
    learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
    min_samples_leaf=40, l2_regularization=1, early_stopping=False, random_state=0
)
hgb_no_env.fit(feat_no_env_tr.values, ytr_s.values)

feat_no_env_va = df_env[no_env_cols]
preds_no_env_raw = hgb_no_env.predict(feat_no_env_va.values)
preds_no_env = _postprocess(preds_no_env_raw, df_env)
result_no_env = np.empty(len(Xva))
result_no_env[sidx] = preds_no_env
rmse_no_env = float(np.sqrt(np.mean((result_no_env - yva.values)**2)))
print(f"RMSE sans enveloppe: {rmse_no_env:.4f}")

print(f"\nGain de l'enveloppe: {rmse_no_env - rmse_full:.4f}")
print(f"Gain attendu (STRATEGIE.md): ~0.64")
