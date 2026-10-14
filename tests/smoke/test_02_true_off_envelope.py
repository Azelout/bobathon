"""
Smoke tests for TrueOffEnvelopeRegressor (experiment 02).

Four properties are verified:
1. A patient's predictions are identical whether it is predicted alone or together
   with other patients (patient isolation).
2. Shuffling the row order does not change predictions.
3. Per-patient predictions are non-decreasing in age and lie in [0, 132].
4. Predicting X_test produces 11,013 finite values with the same index as
   sample_submission.csv.
"""

import numpy as np
import pandas as pd
import pytest

from parkinson import PROJECT_ROOT
from parkinson.estimator import TrueOffEnvelopeRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    """30-patient subset of the training data for fast tests."""
    X = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    pids = X["patient_id"].unique()[:30]
    mask = X["patient_id"].isin(pids)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


@pytest.fixture(scope="module")
def trained_model(small_dataset):
    """Model trained on the first 20 patients of the small dataset."""
    X_all, y_all = small_dataset
    train_pids = X_all["patient_id"].unique()[:20]
    mask = X_all["patient_id"].isin(train_pids)
    X_tr = X_all[mask].reset_index(drop=True)
    y_tr = y_all[mask].reset_index(drop=True)
    model = TrueOffEnvelopeRegressor()
    model.fit(X_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Test 1 — Patient isolation
# ---------------------------------------------------------------------------


def test_patient_isolation(trained_model, small_dataset):
    """Predicting a patient alone vs. together with other patients
    gives the same result.
    """
    X_all, _ = small_dataset
    model = trained_model

    all_pids = X_all["patient_id"].unique()
    # Pick a patient not seen during training (index 20 onward)
    val_pid = all_pids[20]

    X_val_all = X_all[X_all["patient_id"].isin(all_pids[20:25])].reset_index(drop=True)
    X_val_one = X_all[X_all["patient_id"] == val_pid].reset_index(drop=True)

    preds_with_others = model.predict(X_val_all)
    preds_alone = model.predict(X_val_one)

    idx_in_batch = X_val_all[X_val_all["patient_id"] == val_pid].index.to_numpy()
    np.testing.assert_allclose(
        preds_alone,
        preds_with_others[idx_in_batch],
        rtol=1e-5,
        err_msg="Predictions change depending on which other patients are present.",
    )


# ---------------------------------------------------------------------------
# Test 2 — Row-order invariance
# ---------------------------------------------------------------------------


def test_shuffle_invariance(trained_model, small_dataset):
    """Shuffling the row order does not change predictions."""
    X_all, _ = small_dataset
    model = trained_model

    val_pids = X_all["patient_id"].unique()[20:25]
    X_val = X_all[X_all["patient_id"].isin(val_pids)].reset_index(drop=True)

    preds_orig = model.predict(X_val)

    X_shuf = X_val.sample(frac=1, random_state=99).reset_index(drop=True)
    preds_shuf = model.predict(X_shuf)

    # Re-align by original Index column before comparing
    orig_by_idx = {row["Index"]: preds_orig[i] for i, row in X_val.iterrows()}
    expected = np.array([orig_by_idx[row["Index"]] for _, row in X_shuf.iterrows()])

    np.testing.assert_allclose(
        preds_shuf,
        expected,
        rtol=1e-5,
        err_msg="Predictions change when rows are shuffled.",
    )


# ---------------------------------------------------------------------------
# Test 3 — Monotonicity and bounds
# ---------------------------------------------------------------------------


def test_monotone_and_bounds(trained_model, small_dataset):
    """Per-patient predictions are non-decreasing in age and lie in [0, 132]."""
    X_all, _ = small_dataset
    model = trained_model

    val_pids = X_all["patient_id"].unique()[20:]
    X_val = X_all[X_all["patient_id"].isin(val_pids)].reset_index(drop=True)

    preds = model.predict(X_val)

    assert np.all(preds >= 0.0), f"Negative predictions: min={preds.min():.3f}"
    assert np.all(preds <= 132.0), f"Predictions above 132: max={preds.max():.3f}"

    for pid, grp in X_val.groupby("patient_id"):
        p_patient = preds[grp.index.to_numpy()]
        age_order = np.argsort(grp["age"].values)
        diffs = np.diff(p_patient[age_order])
        assert np.all(diffs >= -1e-6), (
            f"Patient {pid}: non-monotone predictions, min diff={diffs.min():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — X_test shape and index
# ---------------------------------------------------------------------------


def test_xtest_predictions():
    """X_test → 11,013 finite predictions with the same index as
    sample_submission.csv.
    """
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    sample_sub = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")

    # Train on a small subset to keep the test fast
    pids = X_train["patient_id"].unique()[:50]
    mask = X_train["patient_id"].isin(pids)
    X_tr = X_train[mask].reset_index(drop=True)
    y_tr = y_train[mask].reset_index(drop=True)

    model = TrueOffEnvelopeRegressor()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_test)

    assert len(preds) == 11_013, f"Wrong number of predictions: {len(preds)}"
    assert np.all(np.isfinite(preds)), "Some predictions are not finite."

    np.testing.assert_array_equal(
        X_test["Index"].values,
        sample_sub["Index"].values,
        err_msg="X_test index does not match sample_submission.csv.",
    )
