import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson.estimator import DebiasWLSRegressor, _build_estimates

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]

# Pli 0 de GroupKFold pour diagnostic rapide
gkf = GroupKFold(5)
folds = list(gkf.split(X, y, groups=X["patient_id"]))
tr_idx, val_idx = folds[0]
X_tr = X.iloc[tr_idx].reset_index(drop=True)
y_tr = y.iloc[tr_idx].reset_index(drop=True)
X_val = X.iloc[val_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

m = DebiasWLSRegressor()
m.fit(X_tr, y_tr)
p = m.predict(X_val)
rmse = np.sqrt(np.mean((p - y_val.values) ** 2))
print(f"RMSE fold 0 (GroupKFold): {rmse:.3f}")
print(
    f"Prior c0_mean={m.prior_['c0_mean']:.1f}"
    f"  c1_mean={m.prior_['c1_mean']:.3f}"
    f"  c2_mean={m.prior_['c2_mean']:.4f}"
)

# Diagnostics sur les estimations (train)
Xs = X_tr.sort_values(["patient_id", "age"]).reset_index(drop=True)
df_est = _build_estimates(Xs, m.factors_)
valid = df_est["est"].notna()
print(
    f"Estimations valides train: {valid.sum()}/{len(df_est)}"
    f" ({100 * valid.mean():.0f}%)"
)
off_pct = df_est["est_off"].notna().mean() * 100
on_pct = df_est["est_on"].notna().mean() * 100
print(
    f"est median={df_est['est'].median():.1f}"
    f"  off_pct={off_pct:.0f}%  on_pct={on_pct:.0f}%"
)

# Distribution des estimations vs cible sur le train
Xs["_y"] = y_tr.values
bias = df_est["est"].values - Xs["_y"].values
ok = np.isfinite(bias)
print(f"Biais moyen estimation train: {bias[ok].mean():.2f}  std: {bias[ok].std():.2f}")

# Diagnostics estimations sur val
Xvs = X_val.sort_values(["patient_id", "age"]).reset_index(drop=True)
df_est_v = _build_estimates(Xvs, m.factors_)
ok_v = df_est_v["est"].notna().values
pv_sorted = p[
    X_val.sort_values(["patient_id", "age"]).index.to_numpy()
    - X_val.sort_values(["patient_id", "age"]).index.min()
]
print(
    f"Estimations valides val: {ok_v.sum()}/{len(df_est_v)} ({100 * ok_v.mean():.0f}%)"
)
