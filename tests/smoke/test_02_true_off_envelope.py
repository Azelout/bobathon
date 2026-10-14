"""
Tests de garde-fous pour TrueOffEnvelopeRegressor (expérience 02).

Quatre propriétés vérifiées :
1. La prédiction d'un patient est identique seul ou avec d'autres patients.
2. Mélanger les lignes ne change pas les prédictions.
3. Les prédictions sont croissantes par patient et comprises entre 0 et 132.
4. La prédiction sur X_test a 11 013 lignes finies avec le même index que
   sample_submission.csv.
"""
import numpy as np
import pandas as pd
import pytest

from parkinson import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Chargement des définitions de l'estimateur sans exécuter les cellules
# ---------------------------------------------------------------------------

_SRC_CACHE: dict = {}


def _load_estimator_defs() -> dict:
    """Parse et exécute uniquement les définitions de l'expérience 02."""
    if _SRC_CACHE:
        return _SRC_CACHE
    src = open(PROJECT_ROOT / "experiments" / "02_true_off_envelope.py").read()
    lines = [l for l in src.split("\n") if not l.startswith("# %%")]
    clean = "\n".join(lines)
    stop = clean.find("# Chargement des données")
    globs: dict = {}
    exec(clean[:stop], globs)
    _SRC_CACHE.update(globs)
    return _SRC_CACHE


# ---------------------------------------------------------------------------
# Fixture : sous-ensemble de patients
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    """Retourne un sous-ensemble de 30 patients pour les tests rapides."""
    X = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    pids = X["patient_id"].unique()[:30]
    mask = X["patient_id"].isin(pids)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


@pytest.fixture(scope="module")
def trained_model(small_dataset):
    """Entraîne le modèle sur le sous-ensemble des 20 premiers patients."""
    X_all, y_all = small_dataset
    defs = _load_estimator_defs()
    TrueOff = defs["TrueOffEnvelopeRegressor"]

    train_pids = X_all["patient_id"].unique()[:20]
    mask = X_all["patient_id"].isin(train_pids)
    X_tr = X_all[mask].reset_index(drop=True)
    y_tr = y_all[mask].reset_index(drop=True)

    model = TrueOff()
    model.fit(X_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Test 1 : isolation par patient
# ---------------------------------------------------------------------------


def test_patient_isolation(trained_model, small_dataset):
    """La prédiction d'un patient est identique seul ou avec d'autres patients."""
    X_all, _ = small_dataset
    model = trained_model

    # Choisir un patient présent dans l'ensemble de validation (non vu à l'entraînement)
    all_pids = X_all["patient_id"].unique()
    val_pid = all_pids[20]  # patient hors du train

    X_val_all = X_all[X_all["patient_id"].isin(all_pids[20:25])].reset_index(drop=True)
    X_val_one = X_all[X_all["patient_id"] == val_pid].reset_index(drop=True)

    preds_with_others = model.predict(X_val_all)
    preds_alone = model.predict(X_val_one)

    # Extraire les prédictions du patient cible dans le batch complet
    idx_in_batch = X_val_all[X_val_all["patient_id"] == val_pid].index.to_numpy()
    preds_patient_in_batch = preds_with_others[idx_in_batch]

    np.testing.assert_allclose(
        preds_alone,
        preds_patient_in_batch,
        rtol=1e-5,
        err_msg="La prédiction change selon les autres patients présents.",
    )


# ---------------------------------------------------------------------------
# Test 2 : invariance à l'ordre des lignes
# ---------------------------------------------------------------------------


def test_shuffle_invariance(trained_model, small_dataset):
    """Mélanger les lignes ne change pas les prédictions."""
    X_all, _ = small_dataset
    model = trained_model

    val_pids = X_all["patient_id"].unique()[20:25]
    X_val = X_all[X_all["patient_id"].isin(val_pids)].reset_index(drop=True)

    preds_orig = model.predict(X_val)

    X_shuf = X_val.sample(frac=1, random_state=99).reset_index(drop=True)
    preds_shuf = model.predict(X_shuf)

    # Réaligner par Index original
    orig_by_idx = {row["Index"]: preds_orig[i] for i, row in X_val.iterrows()}
    expected = np.array([orig_by_idx[row["Index"]] for _, row in X_shuf.iterrows()])

    np.testing.assert_allclose(
        preds_shuf,
        expected,
        rtol=1e-5,
        err_msg="Les prédictions changent quand on mélange les lignes.",
    )


# ---------------------------------------------------------------------------
# Test 3 : croissance et bornes
# ---------------------------------------------------------------------------


def test_monotone_and_bounds(trained_model, small_dataset):
    """Les prédictions sont croissantes par patient et bornées entre 0 et 132."""
    X_all, _ = small_dataset
    model = trained_model

    val_pids = X_all["patient_id"].unique()[20:]
    X_val = X_all[X_all["patient_id"].isin(val_pids)].reset_index(drop=True)

    preds = model.predict(X_val)

    # Bornes globales
    assert np.all(preds >= 0.0), f"Prédictions négatives : min={preds.min():.3f}"
    assert np.all(preds <= 132.0), f"Prédictions > 132 : max={preds.max():.3f}"

    # Monotonie par patient
    for pid, grp in X_val.groupby("patient_id"):
        p_patient = preds[grp.index.to_numpy()]
        # Ordre par âge (le df est déjà trié mais X_val peut avoir un ordre quelconque)
        age_order = np.argsort(grp["age"].values)
        p_sorted = p_patient[age_order]
        diffs = np.diff(p_sorted)
        assert np.all(diffs >= -1e-6), (
            f"Patient {pid}: prédictions non croissantes, diff min={diffs.min():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4 : prédiction sur X_test
# ---------------------------------------------------------------------------


def test_xtest_predictions():
    """X_test → 11 013 lignes finies avec le même index que sample_submission.csv."""
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")["target"]
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")
    sample_sub = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")

    defs = _load_estimator_defs()
    TrueOff = defs["TrueOffEnvelopeRegressor"]

    # Entraîner sur un petit sous-ensemble pour garder le test rapide
    pids = X_train["patient_id"].unique()[:50]
    mask = X_train["patient_id"].isin(pids)
    X_tr = X_train[mask].reset_index(drop=True)
    y_tr = y_train[mask].reset_index(drop=True)

    model = TrueOff()
    model.fit(X_tr, y_tr)

    preds = model.predict(X_test)

    assert len(preds) == 11_013, f"Mauvais nombre de prédictions : {len(preds)}"
    assert np.all(np.isfinite(preds)), "Des prédictions ne sont pas finies."

    # Même index que sample_submission.csv
    np.testing.assert_array_equal(
        X_test["Index"].values,
        sample_sub["Index"].values,
        err_msg="L'index de X_test ne correspond pas à sample_submission.csv.",
    )
