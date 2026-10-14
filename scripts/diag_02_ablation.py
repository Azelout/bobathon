"""
Ablation diagnostic: RMSE with vs without envelope features on one GroupKFold split.

Measures the contribution of the 90-column envelope (block 4) by re-training the
HistGradientBoosting booster without those columns and comparing RMSE to the full model.
Expected gain: ~0.64 RMSE points (see journal/STRATEGIE.md section 4.4).

Usage:
    python scripts/diag_02_ablation.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
from parkinson.estimator import (
    TrueOffEnvelopeRegressor,
    _block1_visit_features,
    _add_on_equivalent,
    _apply_corrections,
    _fit_patient_line,
    _patient_summary_features,
    _envelope_features,
    _postprocess_predictions,
)

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
print("Dataset:", X_train.shape)

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
X_tr = X_train.iloc[tr].reset_index(drop=True)
X_va = X_train.iloc[va].reset_index(drop=True)
y_tr = y_train.iloc[tr].reset_index(drop=True)
y_va = y_train.iloc[va].reset_index(drop=True)
print(f"Train: {X_tr.shape}, Val: {X_va.shape}")

# ── Full model ──────────────────────────────────────────────────────────────
t0 = time.time()
model = TrueOffEnvelopeRegressor()
model.fit(X_tr, y_tr)
print(f"fit: {time.time() - t0:.1f}s")

preds_full = model.predict(X_va)
rmse_full = float(np.sqrt(np.mean((preds_full - y_va.values) ** 2)))
print(f"\nRMSE with envelope:    {rmse_full:.4f}")

# ── Raw booster predictions (no post-processing) ────────────────────────────
sidx = model._sorted_idx(X_va)
X_va_s = X_va.iloc[sidx].reset_index(drop=True)

df = _block1_visit_features(X_va_s)
df = _add_on_equivalent(df, model.drug_curve_)
df = _apply_corrections(df, model.rater_lines_, model.on_effect_, model.on_offset_, model.rater_q95_)
df = _fit_patient_line(df)
df = _patient_summary_features(df)
df_env = _envelope_features(df)

raw_preds = model.booster_.predict(df_env[model.feature_cols_].values)
result_raw = np.empty(len(X_va))
result_raw[sidx] = raw_preds
rmse_raw = float(np.sqrt(np.mean((result_raw - y_va.values) ** 2)))
print(f"RMSE without post-processing: {rmse_raw:.4f}")

# ── Model without envelope columns ──────────────────────────────────────────
env_cols = [c for c in model.feature_cols_ if c.startswith("env_")]
no_env_cols = [c for c in model.feature_cols_ if not c.startswith("env_")]
print(f"\nEnvelope columns: {len(env_cols)} | Remaining: {len(no_env_cols)}")

# Re-train the booster on train features without envelope
sidx_tr = model._sorted_idx(X_tr)
X_tr_s = X_tr.iloc[sidx_tr].reset_index(drop=True)
y_tr_s = pd.Series(y_tr.to_numpy()[sidx_tr])

df_tr = _block1_visit_features(X_tr_s)
df_tr = _add_on_equivalent(df_tr, model.drug_curve_)
df_tr = _apply_corrections(df_tr, model.rater_lines_, model.on_effect_, model.on_offset_, model.rater_q95_)
df_tr = _fit_patient_line(df_tr)
df_tr = _patient_summary_features(df_tr)
df_tr_env = _envelope_features(df_tr)

hgb_no_env = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=600,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1,
    early_stopping=False,
    random_state=0,
)
hgb_no_env.fit(df_tr_env[no_env_cols].values, y_tr_s.values)

preds_no_env_raw = hgb_no_env.predict(df_env[no_env_cols].values)
preds_no_env = _postprocess_predictions(preds_no_env_raw, df_env)
result_no_env = np.empty(len(X_va))
result_no_env[sidx] = preds_no_env
rmse_no_env = float(np.sqrt(np.mean((result_no_env - y_va.values) ** 2)))
print(f"RMSE without envelope: {rmse_no_env:.4f}")

print(f"\nEnvelope gain: {rmse_no_env - rmse_full:.4f} (expected ~0.64)")
