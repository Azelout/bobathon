import numpy as np
import pandas as pd

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["target"]
df = X.copy()
df["target"] = y

# --- Constat 1 : residus parabole vs droite par patient ---
res2, res1 = [], []
for pid, g in df.groupby("patient_id"):
    g = g.sort_values("age")
    t = g["target"].values
    a = g["age"].values - g["age"].mean()
    if len(g) >= 3:
        p2 = np.polyfit(a, t, 2)
        res2.extend((t - np.polyval(p2, a)).tolist())
        p1 = np.polyfit(a, t, 1)
        res1.extend((t - np.polyval(p1, a)).tolist())
    elif len(g) == 2:
        p1 = np.polyfit(a, t, 1)
        res1.extend((t - np.polyval(p1, a)).tolist())
        res2.extend((t - np.polyval(p1, a)).tolist())
    else:
        res2.append(0)
        res1.append(0)
rmse2 = np.sqrt(np.mean(np.array(res2) ** 2))
rmse1 = np.sqrt(np.mean(np.array(res1) ** 2))
print(f"Constat 1 -- RMSE residus parabole: {rmse2:.3f}  droite: {rmse1:.3f}")

# --- Constat 3a : on/target vs time_since_intake_on ---
has_on = (
    df["on"].notna()
    & df["target"].notna()
    & (df["target"] > 3)
    & df["time_since_intake_on"].notna()
)
sub = df[has_on].copy()
sub["ratio_on"] = sub["on"] / sub["target"]
bins = [-0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
sub["tbin"] = pd.cut(sub["time_since_intake_on"], bins=bins)
print("\nConstat 3a -- on/target median par tranche de temps depuis prise:")
print(sub.groupby("tbin", observed=True)["ratio_on"].median().round(3).to_string())

# --- Constat 3b : off/target ---
has_off = df["off"].notna() & df["target"].notna() & (df["target"] > 3)
sub2 = df[has_off].copy()
sub2["ratio_off"] = sub2["off"] / sub2["target"]
off_med = sub2["ratio_off"].median()
off_mea = sub2["ratio_off"].mean()
print(f"\nConstat 3b -- off/target  median: {off_med:.3f}  mean: {off_mea:.3f}")

# --- Constat 3c : biais rater ---
q_off = sub2.groupby("rater_id")["ratio_off"].median()
print(
    f"\nConstat 3c -- ratio off/target par rater: "
    f"min={q_off.min():.3f}  max={q_off.max():.3f}  n_raters={len(q_off)}"
)

sub3 = df[df["on"].notna() & df["target"].notna() & (df["target"] > 3)].copy()
sub3["ratio_on"] = sub3["on"] / sub3["target"]
q_on = sub3.groupby("rater_id")["ratio_on"].median()
print(
    f"Constat 3c -- ratio on/target par rater: "
    f"  min={q_on.min():.3f}  max={q_on.max():.3f}"
)

# Bruit residuel
sub2["resid"] = sub2["off"] - sub2["ratio_off"].median() * sub2["target"]
print(f"\nBruit residuel off (std resid brut): {sub2['resid'].std():.2f}")
sub2["rel_resid"] = sub2["off"] / sub2["target"] - sub2["ratio_off"].median()
print(f"Bruit relatif off (std ratio - median): {sub2['rel_resid'].std():.3f}")
