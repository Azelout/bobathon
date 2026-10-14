"""Diagnostic des colonnes u_off, u_on, u_any."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "pipeline"))
from true_off import visit_features, add_on_equivalent, fit_drug_curve, fit_corrections, patient_features

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
lines = [l for l in src.split("\n") if not l.startswith("# %%")]
clean = "\n".join(lines)
stop = clean.find("# Chargement des données")
globs = {}
exec(clean[:stop], globs)
TrueOff = globs["TrueOffEnvelopeRegressor"]

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
Xtr = X_train.iloc[tr].reset_index(drop=True)
Xva = X_train.iloc[va].reset_index(drop=True)
ytr = y_train.iloc[tr].reset_index(drop=True)
yva = y_train.iloc[va].reset_index(drop=True)

m = TrueOff()
sidx_tr = m._sorted_idx(Xtr)
sidx_va = m._sorted_idx(Xva)
Xtr_s = Xtr.iloc[sidx_tr].reset_index(drop=True)
Xva_s = Xva.iloc[sidx_va].reset_index(drop=True)
ytr_s = pd.Series(ytr.to_numpy()[sidx_tr])

# Référence
d_ref = visit_features(Xtr_s)
curve_ref = fit_drug_curve(d_ref)
d_ref = add_on_equivalent(d_ref, curve_ref)
corr_ref = fit_corrections(d_ref, ytr_s)
d_tr_ref = patient_features(d_ref, corr_ref, (0.0, 1.2, 2.4, 4.0, 6.0), 6)
# Val
d_va_ref_base = visit_features(Xva_s)
d_va_ref_base = add_on_equivalent(d_va_ref_base, curve_ref)
d_va_ref = patient_features(d_va_ref_base, corr_ref, (0.0, 1.2, 2.4, 4.0, 6.0), 6)

# Notre
_b1 = globs["_block1_visit_features"]
_oc = globs["_apply_on_eq"]
_cr = globs["_apply_corrections"]
_pl = globs["_fit_patient_line"]
_ps = globs["_patient_summary_features"]
_ef = globs["_envelope_features"]
_fd = globs["_fit_drug_curve"]
_fs = globs["_fit_patient_sensitivity"]
_ro = globs["_fit_rater_off_line"]
_oe = globs["_fit_on_effect"]
_rq = globs["_fit_rater_q95"]

d_ours_tr = _b1(Xtr_s)
curve = _fd(d_ours_tr)
sens, nboth = _fs(d_ours_tr, curve)
d_ours_tr = _oc(d_ours_tr, curve, sens, nboth)
rl = _ro(d_ours_tr, ytr_s)
oe, oo = _oe(d_ours_tr, ytr_s)
rq = _rq(d_ours_tr, ytr_s)
d_ours_tr = _cr(d_ours_tr, rl, oe, oo, rq)

d_va_ours = _b1(Xva_s)
d_va_ours = _oc(d_va_ours, curve, sens, nboth)
d_va_ours = _cr(d_va_ours, rl, oe, oo, rq)

print("=== u_off (val) ===")
both = d_va_ref["u_off"].notna() & d_va_ours["u_off"].notna()
print(f"N both: {both.sum()}")
diff = (d_va_ref.loc[both, "u_off"] - d_va_ours.loc[both, "u_off"]).abs()
print(f"Max diff u_off: {diff.max():.6f}, mean: {diff.mean():.6f}")

print("\n=== u_on (val) ===")
both_on = d_va_ref["u_on"].notna() & d_va_ours["u_on"].notna()
diff_on = (d_va_ref.loc[both_on, "u_on"] - d_va_ours.loc[both_on, "u_on"]).abs()
print(f"Max diff u_on: {diff_on.max():.6f}, mean: {diff_on.mean():.6f}")

print("\n=== env_a_0_0 vs env_u_any_s0.0_r1 ===")
env_ref = d_va_ref["env_a_0.0_0"]
env_ours_all = _ef(d_va_ours)
env_ours = env_ours_all["env_u_any_s0.0_r1"]
both_e = env_ref.notna() & env_ours.notna()
diff_e = (env_ref.loc[both_e] - env_ours.loc[both_e]).abs()
print(f"N both: {both_e.sum()}")
print(f"Max diff env: {diff_e.max():.4f}, mean: {diff_e.mean():.4f}")

# Quelques valeurs pour un patient
pid = d_va_ours["patient_id"].unique()[0]
mask = d_va_ours["patient_id"] == pid
print(f"\nPatient {pid}:")
print("  u_off ref:", d_va_ref.loc[mask, "u_off"].values)
print("  u_off ours:", d_va_ours.loc[mask, "u_off"].values)
print("  env_a_0_0 ref:", d_va_ref.loc[mask, "env_a_0.0_0"].values)
print("  env_u_any_s0.0_r1 ours:", env_ours_all.loc[mask, "env_u_any_s0.0_r1"].values)
