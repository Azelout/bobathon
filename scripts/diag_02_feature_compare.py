"""Compare les features intermédiaires des deux implémentations sur le même pli."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "pipeline"))
from true_off import visit_features, add_on_equivalent, fit_drug_curve, fit_corrections, patient_features

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
Xtr = X_train.iloc[tr].reset_index(drop=True)
Xva = X_train.iloc[va].reset_index(drop=True)
ytr = y_train.iloc[tr].reset_index(drop=True)
yva = y_train.iloc[va].reset_index(drop=True)

# Référence : features
d_tr_ref = visit_features(Xtr)
curve_ref = fit_drug_curve(d_tr_ref)
d_tr_ref = add_on_equivalent(d_tr_ref, curve_ref)
corr_ref = fit_corrections(d_tr_ref, ytr)
slopes = (0.0, 1.2, 2.4, 4.0, 6.0)
d_tr_ref2 = patient_features(d_tr_ref, corr_ref, slopes, 6)

# Notre implémentation
src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
lines = [l for l in src.split("\n") if not l.startswith("# %%")]
clean = "\n".join(lines)
stop = clean.find("# Chargement des données")
globs = {}
exec(clean[:stop], globs)

TrueOff = globs["TrueOffEnvelopeRegressor"]
m = TrueOff()

# Reproduire le fit interne
sidx = m._sorted_idx(Xtr)
Xtr_s = Xtr.iloc[sidx].reset_index(drop=True)
ytr_s = pd.Series(ytr.to_numpy()[sidx])
d_tr_ours = globs["_block1_visit_features"](Xtr_s)
curve_ours = globs["_fit_drug_curve"](d_tr_ours)
sens_ours, nboth_ours = globs["_fit_patient_sensitivity"](d_tr_ours, curve_ours)
d_tr_ours = globs["_apply_on_eq"](d_tr_ours, curve_ours, sens_ours, nboth_ours)
rater_lines = globs["_fit_rater_off_line"](d_tr_ours, ytr_s)
on_effect, on_offset = globs["_fit_on_effect"](d_tr_ours, ytr_s)
rater_q95 = globs["_fit_rater_q95"](d_tr_ours, ytr_s)
d_tr_ours = globs["_apply_corrections"](d_tr_ours, rater_lines, on_effect, on_offset, rater_q95)

print("=== Comparaison drug_curve ===")
c_ref = curve_ref.to_dict()
c_ours = curve_ours
common_bins = set(c_ref.keys()) & set(c_ours.keys())
print("Bins communs:", len(common_bins), "sur ref:", len(c_ref), "ours:", len(c_ours))
diffs = {k: abs(c_ref[k] - c_ours[k]) for k in common_bins}
print("Max diff curve:", max(diffs.values()) if diffs else "N/A")

print("\n=== Comparaison corrections (rater_line) ===")
rl_ref = corr_ref["rater_line"]
rl_ours = rater_lines
common_raters = set(rl_ref.index) & set(rl_ours.keys())
print("Raters communs:", len(common_raters))
a_diffs = [abs(rl_ref.loc[r, "a"] - rl_ours[r][0]) for r in common_raters]
b_diffs = [abs(rl_ref.loc[r, "b"] - rl_ours[r][1]) for r in common_raters]
print(f"Max diff a: {max(a_diffs):.6f}, Max diff b: {max(b_diffs):.6f}")

print("\n=== Comparaison off_c (validation) ===")
# Appliquer les corrections sur le même ensemble de validation
sidx_va = m._sorted_idx(Xva)
Xva_s = Xva.iloc[sidx_va].reset_index(drop=True)
d_va_ref_base = visit_features(Xva_s)
d_va_ref_base = add_on_equivalent(d_va_ref_base, curve_ref)
d_va_ref2 = patient_features(d_va_ref_base, corr_ref, slopes, 6)

d_va_ours_base = globs["_block1_visit_features"](Xva_s)
d_va_ours_base = globs["_apply_on_eq"](d_va_ours_base, curve_ours, sens_ours, nboth_ours)
d_va_ours_base = globs["_apply_corrections"](d_va_ours_base, rater_lines, on_effect, on_offset, rater_q95)

# Comparer off_c
has_both = d_va_ref2["off_c"].notna() & d_va_ours_base["off_c"].notna()
diff_off_c = (d_va_ref2.loc[has_both, "off_c"] - d_va_ours_base.loc[has_both, "off_c"]).abs()
print(f"Diff off_c (max={diff_off_c.max():.4f}, mean={diff_off_c.mean():.4f})")

has_both_on = d_va_ref2["on_c"].notna() & d_va_ours_base["on_c"].notna()
diff_on_c = (d_va_ref2.loc[has_both_on, "on_c"] - d_va_ours_base.loc[has_both_on, "on_c"]).abs()
print(f"Diff on_c (max={diff_on_c.max():.4f}, mean={diff_on_c.mean():.4f})")

print("\n=== Comparaison u_off ===")
has_uoff = d_va_ref2["u_off"].notna() & d_va_ours_base["u_off"].notna()
diff_uoff = (d_va_ref2.loc[has_uoff, "u_off"] - d_va_ours_base.loc[has_uoff, "u_off"]).abs()
print(f"Diff u_off (max={diff_uoff.max():.4f}, mean={diff_uoff.mean():.4f})")
