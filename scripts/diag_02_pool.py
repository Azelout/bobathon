"""Diagnostic de _pooled_line vs _fit_patient_line."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "pipeline"))
from true_off import visit_features, add_on_equivalent, fit_drug_curve, fit_corrections, patient_features, _pooled_line

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
lines = [l for l in src.split("\n") if not l.startswith("# %%")]
clean = "\n".join(lines)
stop = clean.find("# Chargement des données")
globs = {}
exec(clean[:stop], globs)
TrueOff = globs["TrueOffEnvelopeRegressor"]
_fit_patient_line = globs["_fit_patient_line"]

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
Xtr = X_train.iloc[tr].reset_index(drop=True)
ytr = y_train.iloc[tr].reset_index(drop=True)

our = TrueOff()
sidx = our._sorted_idx(Xtr)
Xtr_s = Xtr.iloc[sidx].reset_index(drop=True)
ytr_s = pd.Series(ytr.to_numpy()[sidx])

# Features intermédiaires
d_ours = globs["_block1_visit_features"](Xtr_s)
curve = globs["_fit_drug_curve"](d_ours)
sens, nboth = globs["_fit_patient_sensitivity"](d_ours, curve)
d_ours = globs["_apply_on_eq"](d_ours, curve, sens, nboth)
rater_lines = globs["_fit_rater_off_line"](d_ours, ytr_s)
on_effect, on_offset = globs["_fit_on_effect"](d_ours, ytr_s)
rater_q95 = globs["_fit_rater_q95"](d_ours, ytr_s)
d_ours = globs["_apply_corrections"](d_ours, rater_lines, on_effect, on_offset, rater_q95)

# Référence
d_ref = visit_features(Xtr_s)
curve_ref = fit_drug_curve(d_ref)
d_ref = add_on_equivalent(d_ref, curve_ref)
corr_ref = fit_corrections(d_ref, ytr_s)
d_ref["off_c"] = (d_ref.off - d_ref.rater_id.map(corr_ref["rater_line"].b)) / d_ref.rater_id.map(corr_ref["rater_line"].a)
d_ref["on_c"] = d_ref.on / np.exp(d_ref.hbin.map(corr_ref["on_effect"]) + d_ref.rater_id.map(corr_ref["on_rater"]).fillna(0))

print("=== off_c identity ===")
both = d_ref["off_c"].notna() & d_ours["off_c"].notna()
print("Max diff off_c:", (d_ref.loc[both, "off_c"] - d_ours.loc[both, "off_c"]).abs().max())

# Appeler les deux fonctions de pooling sur les mêmes données
out_ref, slope_ref, res_ref = _pooled_line(d_ref)
d_ours2 = _fit_patient_line(d_ours)

print("\n=== pool ===")
diff_pool = pd.Series(out_ref) - d_ours2["pool"]
print("Max diff:", diff_pool.abs().max())
print("Mean diff:", diff_pool.abs().mean())

# Comparer sur 1 patient
pid0 = d_ours["patient_id"].unique()[0]
mask_ours = d_ours["patient_id"] == pid0
mask_ref = d_ref["patient_id"] == pid0

print(f"\nPatient {pid0}:")
print("  Ref pool:", out_ref[mask_ref.values])
print("  Ours pool:", d_ours2.loc[mask_ours, "pool"].values)
print("  t_rel ref:", d_ref.loc[mask_ref, "t_rel"].values)
print("  t_rel ours:", d_ours.loc[mask_ours, "t_rel"].values)
print("  off_c ref:", d_ref.loc[mask_ref, "off_c"].values)
print("  off_c ours:", d_ours.loc[mask_ours, "off_c"].values)
print("  on_c ref:", d_ref.loc[mask_ref, "on_c"].values[:4], "...")
print("  on_c ours:", d_ours.loc[mask_ours, "on_c"].values[:4], "...")
