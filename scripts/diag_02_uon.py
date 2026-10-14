"""Diagnostic : u_on diffère entre les deux implémentations."""
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

gkf = GroupKFold(n_splits=5)
tr, va = next(gkf.split(X_train, y_train, groups=X_train["patient_id"]))
Xtr = X_train.iloc[tr].reset_index(drop=True)
Xva = X_train.iloc[va].reset_index(drop=True)
ytr = y_train.iloc[tr].reset_index(drop=True)
yva = y_train.iloc[va].reset_index(drop=True)

m = globs["TrueOffEnvelopeRegressor"]()
sidx_tr = m._sorted_idx(Xtr)
sidx_va = m._sorted_idx(Xva)
Xtr_s = Xtr.iloc[sidx_tr].reset_index(drop=True)
Xva_s = Xva.iloc[sidx_va].reset_index(drop=True)
ytr_s = pd.Series(ytr.to_numpy()[sidx_tr])

# Référence train
d_ref_tr = visit_features(Xtr_s)
c = fit_drug_curve(d_ref_tr)
d_ref_tr = add_on_equivalent(d_ref_tr, c)
corr = fit_corrections(d_ref_tr, ytr_s)
d_tr_ref2 = patient_features(d_ref_tr, corr, (0.0, 1.2, 2.4, 4.0, 6.0), 6)

# Référence val
d_ref_va = visit_features(Xva_s)
d_ref_va = add_on_equivalent(d_ref_va, c)
d_ref_va2 = patient_features(d_ref_va, corr, (0.0, 1.2, 2.4, 4.0, 6.0), 6)

# Nôtre
_b1 = globs["_block1_visit_features"]
_oc = globs["_apply_on_eq"]
_cr = globs["_apply_corrections"]
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

# Trouver les cas où u_on diffère le plus
diff = (d_ref_va2["u_on"] - d_va_ours["u_on"]).abs()
top_idx = diff.nlargest(5).index
print("Top 5 diff u_on:")
for i in top_idx:
    print(f"  Row {i}: ref={d_ref_va2.loc[i,'u_on']:.4f}, ours={d_va_ours.loc[i,'u_on']:.4f}")
    print(f"    on={d_va_ours.loc[i,'on']:.2f}, on_eq_ours={d_va_ours.loc[i,'on_eq']:.4f}")
    print(f"    on_eq_ref={d_ref_va2.loc[i,'on_eq']:.4f}")
    print(f"    q95_ref={d_ref_va2.loc[i,'rater_id']}:", corr["rater_q95"].get(d_ref_va2.loc[i,'rater_id'], "N/A"))
    print(f"    q95_ours:", rq.get(d_ref_va2.loc[i,'rater_id'], "N/A"))
