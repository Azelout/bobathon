"""Compare les prédictions des deux modèles côte à côte (1 pli)."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "pipeline"))
from true_off import TrueOffRegressor

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

# Référence
ref = TrueOffRegressor()
ref.fit(Xtr, ytr)
preds_ref = ref.predict(Xva)
rmse_ref = np.sqrt(np.mean((preds_ref - yva.values)**2))

# Notre modèle
our = TrueOff()
our.fit(Xtr, ytr)
preds_our = our.predict(Xva)
rmse_our = np.sqrt(np.mean((preds_our - yva.values)**2))

print(f"RMSE ref:  {rmse_ref:.4f}")
print(f"RMSE ours: {rmse_our:.4f}")

# Comparer les features passées au booster
sidx = our._sorted_idx(Xva)
Xva_s = Xva.iloc[sidx].reset_index(drop=True)
df_ours = our._build_features(Xva_s)

# Référence : pool et envelope
from true_off import visit_features, add_on_equivalent, patient_features, _envelope
d_va_s = visit_features(Xva_s)
d_va_s = add_on_equivalent(d_va_s, ref.drug_curve_)
d_va_s2 = patient_features(d_va_s, ref.corrections_, ref.envelope_slopes, ref.envelope_ranks)

print("\n=== Comparaison pool ===")
diff_pool = (d_va_s2["pool"] - df_ours["pool"]).abs()
print(f"Max diff pool: {diff_pool.max():.6f}, mean: {diff_pool.mean():.6f}")

print("\n=== Comparaison env_a_0_0 (ref) vs env_u_any_s0.0_r1 (ours) ===")
ref_env_col = "env_a_0.0_0"
our_env_col = "env_u_any_s0.0_r1"
if ref_env_col in d_va_s2.columns and our_env_col in df_ours.columns:
    both = d_va_s2[ref_env_col].notna() & df_ours[our_env_col].notna()
    diff = (d_va_s2.loc[both, ref_env_col] - df_ours.loc[both, our_env_col]).abs()
    print(f"Max diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
else:
    print("Colonnes:", [c for c in d_va_s2.columns if c.startswith("env_a")][:5])
    print("Ours:", [c for c in df_ours.columns if c.startswith("env_u_any")][:5])

print("\n=== Colonnes ref vs ours ===")
print("Ref feature_cols count:", len(ref.columns_))
print("Ours feature_cols count:", len(our.feature_cols_))
# Trouver les colonnes dans la ref mais pas dans ours
ours_set = set(our.feature_cols_)
ref_set = set(ref.columns_)
print("Dans ref mais pas ours:", ref_set - ours_set)
print("Dans ours mais pas ref:", ours_set - ref_set)
