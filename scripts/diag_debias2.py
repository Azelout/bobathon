import pandas as pd
import numpy as np
from parkinson.estimator import DebiasWLSRegressor, _build_estimates, _fit_debias_factors
from sklearn.model_selection import GroupKFold

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]

# Analyser la qualite des estimations individuelles
gkf = GroupKFold(5)
folds = list(gkf.split(X, y, groups=X["patient_id"]))
tr_idx, val_idx = folds[0]
X_tr = X.iloc[tr_idx].reset_index(drop=True)
y_tr = y.iloc[tr_idx].reset_index(drop=True)

factors = _fit_debias_factors(X_tr.sort_values(["patient_id","age"]).reset_index(drop=True),
                               y_tr.reset_index(drop=True))
Xs = X_tr.sort_values(["patient_id","age"]).reset_index(drop=True)
df_est = _build_estimates(Xs, factors)
df_est["_y"] = y_tr.values

ok = df_est["est"].notna() & df_est["_y"].notna()
est_arr = df_est.loc[ok, "est"].values
y_arr = df_est.loc[ok, "_y"].values
resid = est_arr - y_arr
print(f"Residus estimation individuelle: mean={resid.mean():.2f}  std={resid.std():.2f}  RMSE={np.sqrt(np.mean(resid**2)):.3f}")
print(f"Bruit relatif: {resid.std()/y_arr.mean()*100:.1f}%")

# Comparer : est vs off brut vs on brut
has_off = Xs["off"].notna() & (Xs["_y"] > 3)
off_rmse = np.sqrt(np.mean((Xs.loc[has_off, "off"].values - df_est.loc[has_off, "_y"].values)**2))
print(f"RMSE off brut (vs target): {off_rmse:.3f}")
off_est_rmse = np.sqrt(np.mean((df_est.loc[has_off & ok, "est_off"].values - df_est.loc[has_off & ok, "_y"].values)**2))
print(f"RMSE off debiase (vs target): {off_est_rmse:.3f}")

# Combien de mesures par patient dans val ?
X_val = X.iloc[val_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)
nv = X_val.groupby("patient_id").size()
print(f"\nVisites par patient val: mean={nv.mean():.1f}  min={nv.min()}  max={nv.max()}")

# RMSE WLS par nvisites
m = DebiasWLSRegressor()
m.fit(X_tr, y_tr)
p = m.predict(X_val)
df_v = X_val.copy()
df_v["pred"] = p
df_v["target"] = y_val.values
df_v["nv"] = df_v["patient_id"].map(nv)
for grp, g in df_v.groupby(pd.cut(df_v["nv"], bins=[0,3,6,9,20])):
    rmse_g = np.sqrt(np.mean((g["pred"] - g["target"])**2))
    print(f"  nv={grp}: n_rows={len(g)}  RMSE={rmse_g:.3f}")
