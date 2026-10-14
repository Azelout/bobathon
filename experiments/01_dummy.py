# %% [markdown]
# # 01 — Dummy mean baseline
# Always predicts the training-set mean of `target`.
# Sets the error floor: every real model must beat this RMSE (~16.5).

# %%
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

import skore
from parkinson import PROJECT_ROOT
from parkinson.hub import load_skore_credentials

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")

visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[feature_cols]
y = visits["target"]

# %% [markdown]
# ## Fit and evaluate
# `splitter=0.2` → random 80/20 holdout (sufficient for the baseline floor).

# %%
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X, y, splitter=0.2)
report.metrics.rmse()

# %% [markdown]
# ## Push to Skore Hub

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# %% [markdown]
# ## Generate submission CSV

# %%
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
final = clone(dummy).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
submission.shape
# %%
