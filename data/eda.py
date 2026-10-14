# %% [markdown]
# # EDA: Parkinson's Disease UPDRS Score Prediction
#
# Exploratory data analysis of the hackathon clinical dataset (X_train / y_train /
# X_test), run before designing a model.
#
# - **Raw data** is read-only — point `RAW_TRAIN` / `RAW_TARGET` / `RAW_TEST`
#   at the files in `data/`. This file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): one
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data is also in data/.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# X_train holds clinical covariates; y_train holds the continuous target
# (debiased "true OFF" MDS-UPDRS motor score); X_test is the hold-out set.

# %%
RAW_TRAIN = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col=0)
RAW_TARGET = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col=0)
RAW_TEST = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col=0)

# Full train table = features + target for EDA purposes
RAW = RAW_TRAIN.join(RAW_TARGET)

{"train_shape": RAW_TRAIN.shape, "target_shape": RAW_TARGET.shape, "test_shape": RAW_TEST.shape}  # noqa: B018

# %% [markdown]
# ## Table overview — training set (features + target)
#
# Per-column report (types, distributions, missingness) saved to
# `data/eda_train.html`, plus a compact per-column summary.

# %%
report = skrub.TableReport(RAW, title="train (features + target)", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Table overview — test set (features only)
#
# Saved to `data/eda_test.html` for visual inspection.

# %%
report_test = skrub.TableReport(RAW_TEST, title="test (features only)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
test_overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows_test": summary_test.get("n_rows"), "n_columns_test": len(test_overview), "test_columns": test_overview}  # noqa: B018

# %% [markdown]
# ## Target
#
# The target column `target` is the debiased true OFF motor score (regression).
# Distribution summary and skew shape the metric choice and whether to apply
# a target transform.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (temporal validation) and high-cardinality id/group columns
# (grouped validation to avoid leaking patients across folds).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links are
# candidate predictors; implausibly perfect ones are possible leakage flags.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## v2 deep dive — helpers
#
# Everything below works on a copy of the raw tables. `dur` is years since
# diagnosis (`age - age_at_diagnosis`), the main disease-progression clock.

# %%
import numpy as np


def rmse(err):
    err = pd.Series(err).dropna()
    return round(float(np.sqrt((err**2).mean())), 2)


visits = RAW.copy()
visits["dur"] = visits["age"] - visits["age_at_diagnosis"]
visits["visit_no"] = visits.groupby("patient_id").cumcount()

# %% [markdown]
# ## Patients and visits (workbook Q1)
#
# Test patients are disjoint from train, so validation must be grouped by patient.

# %%
per_pt = {name: d.groupby("patient_id").size() for name, d in [("train", RAW_TRAIN), ("test", RAW_TEST)]}
{  # noqa: B018
    "patients": {k: s.size for k, s in per_pt.items()},
    "visits_per_patient": {k: s.describe().round(1).to_dict() for k, s in per_pt.items()},
    "visit_count_histogram_train": per_pt["train"].value_counts().sort_index().to_dict(),
    "patient_overlap": len(set(RAW_TRAIN.patient_id) & set(RAW_TEST.patient_id)),
    "test_raters_unseen": sorted(set(RAW_TEST.rater_id) - set(RAW_TRAIN.rater_id)),
    "rows_sorted_by_age_within_patient": bool(RAW_TRAIN.groupby("patient_id").age.apply(lambda s: s.is_monotonic_increasing).all()),
    "visit_gap_years": RAW_TRAIN.groupby("patient_id").age.diff().describe().round(2).to_dict(),
    "constant_within_patient": {
        c: bool(RAW_TRAIN.groupby("patient_id")[c].nunique(dropna=False).max() == 1)
        for c in ["cohort", "sexM", "gene", "age_at_diagnosis"]
    },
}

# %% [markdown]
# ## Missingness (workbook Q2, Q5)
#
# Missingness follows the clinical story: early visits lack ON and LEDD
# (not yet treated), later visits lack OFF (uncomfortable exam skipped).

# %%
{  # noqa: B018
    "no_off": round(visits.off.isna().mean(), 3),
    "no_on": round(visits.on.isna().mean(), 3),
    "both_missing": int((visits.off.isna() & visits.on.isna()).sum()),
    "both_present": round((visits.off.notna() & visits.on.notna()).mean(), 3),
    "ledd_missing_and_on_present": round((visits.ledd.isna() & visits.on.notna()).mean(), 3),
    "on_missing_and_ledd_present": round((visits.ledd.notna() & visits.on.isna()).mean(), 3),
    "intake_on_present_without_on": int((visits.on.isna() & visits.time_since_intake_on.notna()).sum()),
    "intake_off_present_without_off": int((visits.off.isna() & visits.time_since_intake_off.notna()).sum()),
    "missing_pct_by_cohort": (RAW_TRAIN.groupby("cohort")[["gene", "ledd", "on", "off"]].apply(lambda d: d.isna().mean() * 100).round(1).to_dict(orient="index")),
    "missing_rate_by_visit_no": {
        c: visits.groupby("visit_no")[c].apply(lambda s: s.isna().mean()).round(2).to_dict() for c in ["ledd", "on", "off"]
    },
    "target_by_ledd_missing": visits.groupby(visits.ledd.isna()).agg(target=("target", "mean"), dur=("dur", "mean")).round(2).to_dict(orient="index"),
}

# %% [markdown]
# ## Target and its trajectory (workbook Q3)
#
# Within each patient the target never decreases and is almost exactly a
# quadratic curve in age. The hard part is the per-patient curve, not visit noise.

# %%
fits = []
for _, g in visits.groupby("patient_id"):
    a = g.age.to_numpy() - g.age.iloc[0]
    t = g.target.to_numpy()
    r1 = t - np.polyval(np.polyfit(a, t, 1), a)
    r2 = t - np.polyval(np.polyfit(a, t, 2), a)
    fits.append({"slope_per_year": np.polyfit(a, t, 1)[0], "rmse_linear": rmse(r1), "rmse_quadratic": rmse(r2)})
fits = pd.DataFrame(fits)
{  # noqa: B018
    "target": visits.target.describe().round(2).to_dict(),
    "skew": round(visits.target.skew(), 3),
    "n_target_zero": int((visits.target == 0).sum()),
    "monotone_within_patient": round(visits.groupby("patient_id").target.apply(lambda s: s.is_monotonic_increasing).mean(), 3),
    "per_patient_fit": fits.describe().round(2).to_dict(),
    "target_by_years_since_diagnosis": visits.groupby(pd.cut(visits.dur, [-0.1, 1, 2, 4, 6, 8, 10, 15, 25]), observed=True).target.agg(["mean", "std", "count"]).round(1).to_dict(orient="index"),
}

# %% [markdown]
# ## Measured OFF / ON versus the true OFF (workbook Q4)
#
# `off` is a biased, noisy reading of the target, not a copy. `on` sits
# about 22 points lower because the drug is working.

# %%
off_err = (visits.off - visits.target).dropna()
on_err = (visits.on - visits.target).dropna()
both = visits.dropna(subset=["on", "off"])
corr = {c: round(visits[[c, TARGET]].dropna().corr().iloc[0, 1], 3) for c in ["off", "on", "dur", "age", "ledd", "age_at_diagnosis"]}
{  # noqa: B018
    "pearson_with_target": corr,
    "off_minus_target": off_err.describe().round(2).to_dict(),
    "off_rmse_vs_target": rmse(off_err),
    "on_minus_target": on_err.describe().round(2).to_dict(),
    "on_over_off_ratio": (both.on / both.off.replace(0, np.nan)).describe().round(2).to_dict(),
    "off_error_by_target_bin": visits.assign(e=visits.off - visits.target).dropna(subset=["e"]).groupby(pd.cut(visits.target, [-1, 15, 30, 45, 60, 120]), observed=True).e.agg(["mean", "std"]).round(2).to_dict(orient="index"),
}

# %% [markdown]
# ## Drug timing
#
# OFF readings on treated visits read lower than on untreated ones, but the
# v4 section shows the downward skew also exists in never-treated patients,
# so it is not mainly a residual-drug effect.

# %%
o = visits.dropna(subset=["off"]).assign(e=lambda d: d.off - d.target)
w = visits.dropna(subset=["on", "time_since_intake_on"]).assign(ratio=lambda d: d.on / d.target.replace(0, np.nan))
{  # noqa: B018
    "intake_delay_hours": {c: RAW_TRAIN[c].describe().round(2).to_dict() for c in ["time_since_intake_on", "time_since_intake_off"]},
    "off_error_by_treatment": o.groupby([o.time_since_intake_off.notna().rename("intake_off_known"), o.ledd.notna().rename("ledd_known")]).e.agg(["mean", "std", "count"]).round(2).to_dict(orient="index"),
    "on_over_target_by_hours_since_intake": w.groupby(pd.qcut(w.time_since_intake_on, 6, duplicates="drop"), observed=True).ratio.agg(["median", "count"]).round(2).to_dict(orient="index"),
}

# %% [markdown]
# ## Rater bias
#
# Each rater has a stable, systematic offset. It replicates across disjoint
# patient halves and is shared between that rater's ON and OFF scores.

# %%
rater_bias = o.groupby("rater_id").e.mean()
pts = o.patient_id.unique()
half = set(np.random.default_rng(0).choice(pts, len(pts) // 2, replace=False))
b1 = o[o.patient_id.isin(half)].groupby("rater_id").e.mean()
b2 = o[~o.patient_id.isin(half)].groupby("rater_id").e.mean()
on_bias = visits.assign(e=visits.on - visits.target).dropna(subset=["e"]).groupby("rater_id").e.mean()
{  # noqa: B018
    "n_raters": rater_bias.size,
    "raters_per_patient": RAW_TRAIN.groupby("patient_id").rater_id.nunique().describe().round(1).to_dict(),
    "off_bias_per_rater": rater_bias.describe().round(2).to_dict(),
    "most_negative": rater_bias.nsmallest(3).round(2).to_dict(),
    "most_positive": rater_bias.nlargest(3).round(2).to_dict(),
    "split_half_corr": round(b1.corr(b2), 3),
    "on_vs_off_bias_corr": round(on_bias.corr(rater_bias), 3),
}

# %% [markdown]
# ## Simple reference errors
#
# Not models — just how far the obvious proxies sit from the target.
# Naive per-patient pooling of `off` is *worse* than the raw reading.

# %%
patient_mean_off = visits.groupby("patient_id").off.transform("mean")
{  # noqa: B018
    "dummy_mean_rmse": rmse(visits.target - visits.target.mean()),
    "raw_off_rmse_where_present": rmse(visits.off - visits.target),
    "patient_mean_off_rmse": rmse(patient_mean_off - visits.target),
    "train_vs_test_means": {c: (round(RAW_TRAIN[c].mean(), 2), round(RAW_TEST[c].mean(), 2)) for c in ["age", "ledd", "on", "off", "time_since_intake_on"]},
}

# %% [markdown]
# ## v3 — measurement mechanism
#
# How measured OFF / ON derive from the true OFF. Corrections below are fitted
# in-sample with the target: this is a diagnostic of what pooling *can* reach,
# not a model score.

# %%
offv = visits.dropna(subset=["off"])
clean = offv[np.log(offv.off.clip(lower=0.5) / offv.target.replace(0, np.nan)) > -1].copy()
clean["e"] = clean.off - 0.88 * clean.target
clean["e"] -= clean.groupby("rater_id").e.transform("mean")
clean["e_prev"] = clean.groupby("patient_id").e.shift()
k = clean.groupby("patient_id").e.transform("size")
pm = clean.groupby("patient_id").e.transform("mean")
onv = visits.dropna(subset=["on"]).query("target > 5").copy()
onv["log_ratio"] = np.log(onv.on.clip(lower=0.5) / onv.target)
timed = onv.dropna(subset=["time_since_intake_on"])
{  # noqa: B018
    "scores_are_integers": bool((visits.off.dropna() % 1 == 0).all() and (visits.on.dropna() % 1 == 0).all()),
    "off_zero_rows": int((visits.off == 0).sum()),
    "off_vs_target_linear_fit": np.polyfit(offv.target, offv.off, 1).round(3).tolist(),
    "per_rater_off_over_target": offv[offv.target > 5].groupby("rater_id").apply(lambda g: (g.off / g.target).median()).describe().round(3).to_dict(),
    "off_error_lag1_corr_within_patient": round(clean[["e", "e_prev"]].dropna().corr().iloc[0, 1], 3),
    "off_error_between_patient_share": round(pm[k >= 3].var() / clean.e[k >= 3].var(), 3),
    "share_expected_if_independent": round((1 / k[k >= 3]).mean(), 3),
    "off_error_sd_by_target": clean.groupby(pd.cut(clean.target, [0, 15, 30, 45, 60, 120]), observed=True).e.std().round(2).to_dict(),
    "on_log_ratio_by_hours_since_intake": timed.groupby(pd.cut(timed.time_since_intake_on, [-0.01, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 7]), observed=True).log_ratio.agg(["mean", "std", "count"]).round(3).to_dict(orient="index"),
    "on_log_ratio_when_timing_missing": onv[onv.time_since_intake_on.isna()].log_ratio.agg(["mean", "std"]).round(3).to_dict(),
}

# %% [markdown]
# ## v3 — patient curves and pooling diagnostic
#
# Patient covariates barely predict the curve, so information must come from
# the patient's own measurements. Bias-corrected per-patient linear pooling
# shows how much that is worth.

# %%
rows = []
for _, g in visits.groupby("patient_id"):
    a = g.age.to_numpy() - g.age.iloc[0]
    slope, t0 = np.polyfit(a, g.target.to_numpy(), 1)
    rows.append({"t0": g.target.iloc[0], "slope": slope, "dur0": g.dur.iloc[0], "ad": g.age_at_diagnosis.iloc[0], "sexM": g.sexM.iloc[0], "cohort": g.cohort.iloc[0], "gene": g.gene.fillna("NA").iloc[0], "n_visits": len(g)})
curves = pd.DataFrame(rows).dropna(subset=["dur0"])
curves["t_diag"] = curves.t0 - curves.slope * curves.dur0


def r2(target, frame):
    A = np.column_stack([np.ones(len(frame)), pd.get_dummies(frame, drop_first=True).astype(float).to_numpy()])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return round(1 - (target - A @ coef).var() / target.var(), 3)


cov = curves[["ad", "sexM", "cohort", "gene"]]
rater_fit = offv.groupby("rater_id").apply(lambda g: pd.Series(np.polyfit(g.target, g.off, 1), index=["a", "b"]))
pv = visits.copy()
pv["off_c"] = (pv.off - pv.rater_id.map(rater_fit.b)) / pv.rater_id.map(rater_fit.a)
bins = [-2, -0.5, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 7]
onv["bin"] = pd.cut(onv.time_since_intake_on.fillna(-1), bins)
effect = onv.groupby("bin", observed=True).log_ratio.median()
rater_on = (onv.log_ratio - onv.bin.map(effect).astype(float)).groupby(onv.rater_id).median()
pv["on_c"] = pv.on / np.exp(pd.cut(pv.time_since_intake_on.fillna(-1), bins).map(effect).astype(float) + pv.rater_id.map(rater_on))


def pooled(g, deg):
    a = np.r_[g.age[g.off_c.notna()], g.age[g.on_c.notna()]]
    m = np.r_[g.off_c.dropna(), g.on_c.dropna()]
    w = np.r_[np.ones(g.off_c.notna().sum()), 0.6 * np.ones(g.on_c.notna().sum())]
    if len(np.unique(a)) <= deg:
        return pd.Series(np.average(m, weights=w), index=g.index)
    return pd.Series(np.polyval(np.polyfit(a, m, deg, w=np.sqrt(w)), g.age), index=g.index)


pool = {deg: pv.groupby("patient_id", group_keys=False).apply(lambda g, d=deg: pooled(g, d)) for deg in (0, 1, 2)}
neg_slope = pv.assign(p=pool[1]).groupby("patient_id").apply(lambda g: np.polyfit(g.age, g.p, 1)[0] < 0).mean()
{  # noqa: B018
    "r2_level_at_diagnosis_from_covariates": r2(curves.t_diag.to_numpy(), cov),
    "r2_slope_from_covariates": r2(curves.slope.to_numpy(), cov),
    "share_4_visit_patients_by_cohort": curves.groupby("cohort").n_visits.apply(lambda s: (s == 4).mean()).round(2).to_dict(),
    "rmse_corrected_off_visit_level": rmse(pv.off_c - pv.target),
    "rmse_corrected_on_visit_level": rmse(pv.on_c - pv.target),
    "rmse_pooled_constant": rmse(pool[0] - pv.target),
    "rmse_pooled_linear": rmse(pool[1] - pv.target),
    "rmse_pooled_linear_off_absent": rmse((pool[1] - pv.target)[pv.off.isna()]),
    "rmse_pooled_quadratic": rmse(pool[2] - pv.target),
    "share_pooled_negative_slopes": round(float(neg_slope), 3),
}

# %% [markdown]
# ## v4 — corrections after the independent review
#
# 1. The OFF error is one-sided: a sharp upper edge per rater and a long
#    downward tail, present even in patients who are never treated.
# 2. ON is best read relative to OFF at the same visit: given intake time,
#    log(on/off) is nearly constant within a patient. Nothing here uses the
#    target except the rater edge and the skew check.

# %%
offr = visits.dropna(subset=["off"]).query("target > 15").copy()
offr["ratio"] = offr.off / offr.target
offr["u"] = offr.ratio / offr.groupby("rater_id").ratio.transform(lambda s: s.quantile(0.95))
treated_any = (visits.on.notna() | visits.ledd.notna() | visits.time_since_intake_on.notna()).groupby(visits.patient_id).any()
never = offr.patient_id.map(treated_any) == False  # noqa: E712
u_res = offr.u - offr.groupby("rater_id").u.transform("median")

HB = [-2, -0.5, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 7]
both = visits.dropna(subset=["on", "off"]).query("off >= 8 and on >= 1").copy()
both["log_on_off"] = np.log(both.on / both.off)
both["hbin"] = pd.cut(both.time_since_intake_on.fillna(-1), HB)
both["resid"] = both.log_on_off - both.groupby("hbin", observed=True).log_on_off.transform("median")
n_both = both.groupby("patient_id").resid.transform("size")
pt_mean = both.groupby("patient_id").resid.transform("mean")
both["odd"] = both.groupby("patient_id").cumcount() % 2
halves = both[n_both >= 4].groupby(["patient_id", "odd"]).resid.mean().unstack()
{  # noqa: B018
    "off_skew_within_rater": round(float(u_res.skew()), 2),
    "u_quantiles_all": offr.u.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).to_dict(),
    "u_median_never_treated": round(float(offr.u[never].median()), 3),
    "u_median_treated_patients": round(float(offr.u[~never].median()), 3),
    "n_never_treated_off_readings": int(never.sum()),
    "log_on_off_by_hours_since_intake": both.groupby("hbin", observed=True).log_on_off.median().round(2).to_dict(),
    "log_on_off_resid_sd_all": round(float(both.resid.std()), 3),
    "log_on_off_resid_sd_within_patient": round(float((both.resid - pt_mean)[n_both >= 2].std()), 3),
    "between_patient_share": round(float(pt_mean[n_both >= 3].var() / both.resid[n_both >= 3].var()), 3),
    "share_expected_by_chance": round(float((1 / n_both[n_both >= 3]).mean()), 3),
    "odd_even_visit_agreement": round(float(halves.corr().iloc[0, 1]), 3),
    "share_visits_on_only": round(float((visits.on.notna() & visits.off.isna()).mean()), 3),
}

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
