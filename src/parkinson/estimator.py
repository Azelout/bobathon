"""
Estimateur True OFF à enveloppe pour la prédiction du score MDS-UPDRS III OFF.

Implémente les six blocs de la stratégie décrite dans journal/STRATEGIE.md :
1. Variables de visite et de traitement (sans cible)
2. Conversion de `on` en équivalent `off` (sans cible, courbe apprise dans fit)
3. Corrections par évaluateur (apprises dans fit)
4. Variables par patient : droite pondérée + résumés + enveloppe 90 colonnes
5. HistGradientBoostingRegressor
6. Post-traitement par patient : parabole → isotonique → clip [0, 132]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constantes de discrétisation
# ---------------------------------------------------------------------------

# Tranches de délai depuis la prise ON, coarses (pour effet ON vs cible)
HBIN_EDGES = [-2.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 7.0]

# Tranches fines (pour la courbe médicament off/on)
HB_EDGES = [-2.0, -0.5, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0]

# Valeur sentinelle pour les délais manquants
DELAY_FILL = -1.0

# Pentes pour l'enveloppe (en points / an)
ENVELOPE_SLOPES = [0.0, 1.2, 2.4, 4.0, 6.0]

# Nombre de rangs conservés par série et par pente
ENVELOPE_RANKS = 6

# Séries d'enveloppe
ENVELOPE_SERIES = ["u_off", "u_on", "u_any"]

# Encodage fixe des mutations génétiques
GENE_MAP = {"GBA+": 0, "LRRK2+": 1, "No Mutation": 2, "OTHER+": 3}


def _assign_bin(series: pd.Series, edges: list[float]) -> pd.Series:
    """Discrétise une série en indices de tranches (entiers 0..n-1)."""
    return pd.cut(
        series,
        bins=edges,
        labels=False,
        right=True,
        include_lowest=True,
    ).astype("Int64")


# ---------------------------------------------------------------------------
# Bloc 1 — Variables de visite et de traitement (sans cible)
# ---------------------------------------------------------------------------


def _block1_visit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les variables de visite et de traitement sans utiliser la cible."""
    out = df.copy()

    # Rang et âge minimal/maximal par patient
    patient_first_age = df.groupby("patient_id")["age"].transform("min")
    patient_last_age = df.groupby("patient_id")["age"].transform("max")

    out["t_rel"] = df["age"] - patient_first_age
    out["vis"] = df.groupby("patient_id").cumcount()
    out["nv"] = df.groupby("patient_id")["age"].transform("count")
    out["span"] = patient_last_age - patient_first_age

    # Durée depuis le diagnostic
    out["dur"] = df["age"] - df["age_at_diagnosis"]

    # Encodages catégoriels fixes
    out["rater"] = (
        df["rater_id"]
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .astype("Int64")
    )
    out["coh"] = (df["cohort"] == "B").astype(int)
    out["gene_c"] = df["gene"].map(GENE_MAP).fillna(-1).astype(int)

    # Tranches de délai ON (sentinelle -1 pour les manquants)
    delay_on = df["time_since_intake_on"].fillna(DELAY_FILL)
    out["hbin"] = _assign_bin(delay_on, HBIN_EDGES)
    out["hb"] = _assign_bin(delay_on, HB_EDGES)

    # Statut traité : 1 si on, ledd ou délai ON présent
    treat_mask = (
        df["on"].notna() | df["ledd"].notna() | df["time_since_intake_on"].notna()
    )
    out["treat"] = treat_mask.astype(int)

    # Âge au premier traitement par patient
    treated = df[treat_mask][["patient_id", "age"]]
    first_treat = treated.groupby("patient_id")["age"].min().rename("age_first_treat")
    out = out.join(first_treat, on="patient_id")

    out["t_since_treat"] = df["age"] - out["age_first_treat"]

    # Dernière visite non traitée par patient
    untreated = df[~treat_mask][["patient_id", "age"]]
    last_untr = untreated.groupby("patient_id")["age"].max().rename("age_last_untr")
    out = out.join(last_untr, on="patient_id")
    out["t_to_lastuntr"] = df["age"] - out["age_last_untr"]

    out["dur_treat"] = out["age_first_treat"] - df["age_at_diagnosis"]

    out["n_treat"] = df.groupby("patient_id")["on"].transform(
        lambda s: s.notna().sum()
    )

    return out


# ---------------------------------------------------------------------------
# Bloc 2 — Conversion de `on` (sans cible, appris dans fit)
# ---------------------------------------------------------------------------


def _fit_drug_curve(df: pd.DataFrame) -> dict:
    """
    Apprend la courbe d'effet du médicament (médiane de log(on/off) par tranche hb)
    sur les visites doubles (off ≥ 8 et on ≥ 1).

    Retourne un dict {bin_index: median_log_ratio}.
    """
    doubles = df[df["off"].notna() & df["on"].notna() & (df["off"] >= 8) & (df["on"] >= 1)].copy()
    doubles["log_ratio"] = np.log(doubles["on"] / doubles["off"])
    curve = doubles.groupby("hb")["log_ratio"].median().to_dict()
    return curve


def _add_on_equivalent(df: pd.DataFrame, curve: dict, lam: float = 1.0) -> pd.DataFrame:
    """
    Calcule la sensibilité par patient (shrunk mean residual) et on_eq = on * exp(-(courbe+sens)).

    La sensibilité est calculée à la volée sur les données passées — les patients de
    validation utilisent donc leurs propres visites doubles (sans cible, légitime selon P2).
    """
    doubles = df[
        df["off"].notna() & df["on"].notna() & (df["off"] >= 8) & (df["on"] >= 1)
    ].copy()
    doubles["log_ratio"] = np.log(doubles["on"] / doubles["off"])
    doubles["curve_val"] = doubles["hb"].map(curve).fillna(0.0)
    doubles["residual"] = doubles["log_ratio"] - doubles["curve_val"]

    agg = doubles.groupby("patient_id")["residual"].agg(["sum", "count"])
    sens_map = (agg["sum"] / (agg["count"] + lam)).to_dict()
    nboth_map = agg["count"].to_dict()

    out = df.copy()
    out["sens"] = df["patient_id"].map(sens_map).fillna(0.0)
    out["nboth"] = df["patient_id"].map(nboth_map).fillna(0).astype(int)
    out["on_eq"] = df["on"] * np.exp(-(df["hb"].map(curve).fillna(0.0) + out["sens"]))
    return out


# ---------------------------------------------------------------------------
# Bloc 3 — Corrections avec la cible (apprises dans fit)
# ---------------------------------------------------------------------------


def _fit_rater_off_line(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Ajuste off = a*target + b par évaluateur (via régression linéaire simple).

    Retourne {rater_id: (a, b)}.
    """
    joined = df[["rater_id", "off"]].copy()
    joined["target"] = y.values
    has_off = joined["off"].notna()
    rater_lines = {}
    for rater, grp in joined[has_off].groupby("rater_id"):
        if len(grp) < 2:
            rater_lines[rater] = (1.0, 0.0)
            continue
        t = grp["target"].values
        o = grp["off"].values
        # Régression OLS : off = a*target + b
        a, b = np.polyfit(t, o, 1)
        if abs(a) < 1e-6:
            a = 1e-6
        rater_lines[rater] = (float(a), float(b))
    return rater_lines


def _fit_on_effect(df: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    """
    Effet ON : médiane de log(max(on,0.5)/target) par tranche hbin.
    Décalage par évaluateur : médiane des écarts à cet effet.

    Retourne (effect_by_hbin, offset_by_rater).
    """
    joined = df[["rater_id", "on", "hbin"]].copy()
    joined["target"] = y.values
    has_on = joined["on"].notna() & (joined["target"] > 3)
    sub = joined[has_on].copy()
    sub["log_ratio"] = np.log(np.maximum(sub["on"], 0.5) / sub["target"])
    effect = sub.groupby("hbin")["log_ratio"].median().to_dict()

    sub["expected"] = sub["hbin"].map(effect).fillna(0.0)
    sub["residual"] = sub["log_ratio"] - sub["expected"]
    offset = sub.groupby("rater_id")["residual"].median().to_dict()
    return effect, offset


def _fit_rater_q95(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    95e centile de off/target par évaluateur, sur les visites avec off et target > 15.
    """
    joined = df[["rater_id", "off"]].copy()
    joined["target"] = y.values
    valid = joined[joined["off"].notna() & (joined["target"] > 15)].copy()
    valid["ratio"] = valid["off"] / valid["target"]
    q95 = valid.groupby("rater_id")["ratio"].quantile(0.95).to_dict()
    return q95


def _apply_corrections(
    df: pd.DataFrame,
    rater_lines: dict,
    on_effect: dict,
    on_offset: dict,
    q95: dict,
) -> pd.DataFrame:
    """Calcule off_c, on_c, u_off, u_on, u_any."""
    out = df.copy()

    # off_c
    a_vals = df["rater_id"].map({r: v[0] for r, v in rater_lines.items()}).fillna(1.0)
    b_vals = df["rater_id"].map({r: v[1] for r, v in rater_lines.items()}).fillna(0.0)
    out["off_c"] = np.where(
        df["off"].notna(),
        (df["off"] - b_vals) / a_vals,
        np.nan,
    )

    # on_c
    eff = df["hbin"].map(on_effect).fillna(0.0)
    off_rater = df["rater_id"].map(on_offset).fillna(0.0)
    out["on_c"] = np.where(
        df["on"].notna(),
        df["on"] / np.exp(eff + off_rater),
        np.nan,
    )

    # Normalisation par le bord de l'évaluateur
    q = df["rater_id"].map(q95).fillna(1.0)
    out["u_off"] = np.where(df["off"].notna(), df["off"] / q, np.nan)
    out["u_on"] = np.where(df["on_eq"].notna(), df["on_eq"] / q, np.nan)
    # u_any : u_off en priorité, sinon u_on
    out["u_any"] = np.where(out["u_off"].notna(), out["u_off"], out["u_on"])

    return out


# ---------------------------------------------------------------------------
# Bloc 4 — Variables par patient
# ---------------------------------------------------------------------------


def _fit_patient_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajuste une droite pondérée par patient sur off_c et on_c en fonction de t_rel centré.

    Chaque mesure disponible (off_c ou on_c) contribue indépendamment. Une visite avec
    les deux mesures compte deux fois.

    Sorties : pool (valeur de la droite à chaque visite), pslope, pres.
    """
    out = df.copy()
    pool = np.full(len(df), np.nan)
    pslope = np.full(len(df), np.nan)
    pres = np.full(len(df), np.nan)

    LAM = 1.0 / 1.35 ** 2   # pénalité de pente
    B0 = 2.5                  # a priori de pente

    for pid, grp in df.groupby("patient_id"):
        idx = grp.index.to_numpy()
        t = grp["t_rel"].to_numpy()
        tc = t - t.mean()
        mo = grp["off_c"].to_numpy()
        mn = grp["on_c"].to_numpy()

        # Regrouper toutes les mesures disponibles (off_c + on_c séparément)
        T = np.concatenate([tc[~np.isnan(mo)], tc[~np.isnan(mn)]])
        M = np.concatenate([mo[~np.isnan(mo)], mn[~np.isnan(mn)]])

        if len(M) == 0:
            continue

        level = float(np.nanmedian(M))
        level = max(level, 5.0)

        # Poids différenciés : off_c moins bruité que on_c
        W = np.concatenate([
            np.full((~np.isnan(mo)).sum(), 1.0 / (2.0 + 0.17 * level) ** 2),
            np.full((~np.isnan(mn)).sum(), 1.0 / (0.45 * level) ** 2),
        ])

        # Système normal avec pénalité sur la pente
        sw = W.sum()
        swt = (W * T).sum()
        swt2 = (W * T ** 2).sum()
        swv = (W * M).sum()
        swvt = (W * T * M).sum()
        w_mean = W.mean()

        A = np.array([
            [sw, swt],
            [swt, swt2 + LAM * w_mean],
        ])
        rhs = np.array([swv, swvt + LAM * w_mean * B0])
        a, b = np.linalg.solve(A + 1e-9 * np.eye(2), rhs)

        # Pente bornée à 0
        b = max(b, 0.0)
        if b == 0.0:
            a = swv / sw

        pool[idx] = a + b * tc
        pslope[idx] = b
        pres[idx] = float(np.sqrt(np.average((M - (a + b * T)) ** 2, weights=W)))

    out["pool"] = pool
    out["pslope"] = pslope
    out["pres"] = pres
    return out


def _patient_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule n_off, n_on, médiane de off_c/on_c, off_c de la visite précédente/suivante."""
    out = df.copy()
    g = df.groupby("patient_id")

    out["n_off"] = g["off_c"].transform(lambda s: s.notna().sum())
    out["n_on"] = g["on_c"].transform(lambda s: s.notna().sum())
    out["offc_pmed"] = g["off_c"].transform("median")
    out["onc_pmed"] = g["on_c"].transform("median")

    # off_c de la visite précédente / suivante via ffill/bfill
    out["offc_prev"] = g["off_c"].transform(lambda s: s.ffill().shift())
    out["offc_next"] = g["off_c"].transform(lambda s: s.bfill().shift(-1))

    return out


def _envelope_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 90 colonnes d'enveloppe.

    Pour chaque visite v, série (u_off, u_on, u_any) et pente s :
    u_j + s*(age_v - age_j) pour chaque mesure j du patient,
    triées décroissantes, rangs 1 à 6.
    """
    n = len(df)
    slopes = np.array(ENVELOPE_SLOPES)  # shape (S,)
    n_s = len(slopes)
    n_r = ENVELOPE_RANKS
    n_ser = len(ENVELOPE_SERIES)

    # Résultat final : (n_rows, n_ser * n_s * n_r)
    env = np.full((n, n_ser * n_s * n_r), np.nan)

    for si, ser in enumerate(ENVELOPE_SERIES):
        c_off = si * n_s * n_r

        for pid, grp in df.groupby("patient_id"):
            vidx = grp.index.to_numpy()
            ages = grp["age"].values
            vals = grp[ser].values

            valid_mask = ~np.isnan(vals)
            if valid_mask.sum() == 0:
                continue

            valid_ages = ages[valid_mask]
            valid_vals = vals[valid_mask]
            M = len(valid_vals)

            # age_diff[v, j] = age_v - age_j  →  shape (V, M)
            age_diff = ages[:, None] - valid_ages[None, :]

            # projected[v, slope_i, j] = valid_vals[j] + slopes[slope_i] * age_diff[v,j]
            projected = (
                valid_vals[None, None, :]
                + slopes[None, :, None] * age_diff[:, None, :]
            )  # shape (V, S, M)

            sorted_proj = np.sort(projected, axis=2)[:, :, ::-1]  # shape (V, S, M)

            for si2 in range(n_s):
                for r in range(n_r):
                    col_idx = c_off + si2 * n_r + r
                    if r < M:
                        env[vidx, col_idx] = sorted_proj[:, si2, r]

    col_names = [
        f"env_{ser}_s{s:.1f}_r{rank}"
        for ser in ENVELOPE_SERIES
        for s in ENVELOPE_SLOPES
        for rank in range(1, ENVELOPE_RANKS + 1)
    ]
    env_df = pd.DataFrame(env, index=df.index, columns=col_names)
    return pd.concat([df, env_df], axis=1)


# ---------------------------------------------------------------------------
# Bloc 6 — Post-traitement par patient
# ---------------------------------------------------------------------------


def _postprocess_predictions(preds: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Post-traitement par patient (sur les prédictions triées par âge) :
    1. Parabole en âge centré (degré min(2, nv-1)).
    2. IsotonicRegression.
    3. Clip [0, 132].
    """
    result = preds.copy()

    for pid, grp in df.groupby("patient_id"):
        idx = grp.index.to_numpy()
        age = grp["age"].values
        p = preds[idx]
        nv = len(idx)

        # 1. Projection parabolique
        degree = min(2, nv - 1)
        if degree >= 1:
            age_c = age - age.mean()
            try:
                coeffs = np.polyfit(age_c, p, degree)
                p = np.polyval(coeffs, age_c)
            except (np.linalg.LinAlgError, ValueError):
                pass

        # 2. Isotonique
        if nv > 1:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            p = iso.fit_transform(age, p)

        # 3. Bornes
        p = np.clip(p, 0.0, 132.0)
        result[idx] = p

    return result


# ---------------------------------------------------------------------------
# Listes de colonnes fixes pour le booster
# ---------------------------------------------------------------------------

BASE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "on", "off",
    "dur", "t_rel", "vis", "nv", "span",
    "rater", "coh", "gene_c",
]

POOL_COLS = [
    "off_c", "on_c", "pool", "pslope", "pres",
    "n_off", "n_on", "offc_pmed", "onc_pmed",
    "offc_prev", "offc_next",
]

TREAT_COLS = [
    "on_eq", "nboth", "treat", "sens",
    "t_since_treat", "t_to_lastuntr", "dur_treat", "n_treat",
]


# ---------------------------------------------------------------------------
# Estimateur principal
# ---------------------------------------------------------------------------


class TrueOffEnvelopeRegressor(BaseEstimator, RegressorMixin):
    """
    Estimateur scikit-learn implémentant la stratégie True OFF à enveloppe.

    Blocs 1 à 6 de journal/STRATEGIE.md. Trie les lignes en interne par
    patient_id puis age et restitue l'ordre d'origine. Tout ce qui utilise
    la cible est appris dans fit.

    Parameters
    ----------
    learning_rate : float
        Taux d'apprentissage du gradient boosting.
    max_iter : int
        Nombre d'itérations du gradient boosting.
    max_leaf_nodes : int
        Nombre maximal de feuilles par arbre.
    min_samples_leaf : int
        Nombre minimal d'échantillons par feuille.
    l2_regularization : float
        Régularisation L2.
    random_state : int
        Graine aléatoire.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 600,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _sorted_idx(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les indices positionnels de X triés par patient_id puis age."""
        return (
            X.reset_index(drop=True)
            .sort_values(["patient_id", "age"], kind="stable")
            .index.to_numpy()
        )

    def _build_features(self, X_sorted: pd.DataFrame) -> pd.DataFrame:
        """Construit toutes les variables à partir du df trié (blocs 1 à 4)."""
        df = _block1_visit_features(X_sorted)
        df = _add_on_equivalent(df, self.drug_curve_)
        df = _apply_corrections(
            df,
            self.rater_lines_,
            self.on_effect_,
            self.on_offset_,
            self.rater_q95_,
        )
        df = _fit_patient_line(df)
        df = _patient_summary_features(df)
        df = _envelope_features(df)
        return df

    def _select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list, int | None]:
        """Sélectionne les colonnes pour le booster selon les listes fixes."""
        env_cols = [c for c in df.columns if c.startswith("env_")]
        feature_cols = BASE_COLS + POOL_COLS + TREAT_COLS + env_cols

        # Garder uniquement les colonnes présentes dans df
        feature_cols = [c for c in feature_cols if c in df.columns]

        rater_idx = feature_cols.index("rater") if "rater" in feature_cols else None

        return df[feature_cols], feature_cols, rater_idx

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrueOffEnvelopeRegressor":
        """Apprend les corrections et le booster sur les données d'entraînement."""
        sidx = self._sorted_idx(X)
        X_sorted = X.iloc[sidx].reset_index(drop=True)

        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_sorted = pd.Series(y_arr[sidx], name="target")

        # Bloc 1
        df = _block1_visit_features(X_sorted)

        # Bloc 2 : courbe médicament apprise sur le pli d'entraînement (sans cible)
        self.drug_curve_ = _fit_drug_curve(df)
        df = _add_on_equivalent(df, self.drug_curve_)

        # Bloc 3 : corrections avec la cible
        self.rater_lines_ = _fit_rater_off_line(df, y_sorted)
        self.on_effect_, self.on_offset_ = _fit_on_effect(df, y_sorted)
        self.rater_q95_ = _fit_rater_q95(df, y_sorted)
        df = _apply_corrections(
            df,
            self.rater_lines_,
            self.on_effect_,
            self.on_offset_,
            self.rater_q95_,
        )

        # Bloc 4
        df = _fit_patient_line(df)
        df = _patient_summary_features(df)
        df = _envelope_features(df)

        # Bloc 5
        feat_df, self.feature_cols_, rater_idx = self._select_features(df)
        cat_features = [rater_idx] if rater_idx is not None else []

        self.booster_ = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            categorical_features=cat_features if cat_features else None,
            random_state=self.random_state,
        )
        self.booster_.fit(feat_df.values, y_sorted.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit en triant en interne, puis restitue l'ordre d'origine."""
        sidx = self._sorted_idx(X)
        X_sorted = X.iloc[sidx].reset_index(drop=True)

        df = self._build_features(X_sorted)
        feat_df = df[self.feature_cols_]
        raw_preds = self.booster_.predict(feat_df.values)

        # Bloc 6 : post-traitement par patient (df est déjà trié par patient+age)
        preds_postproc = _postprocess_predictions(raw_preds, df)

        # Restituer l'ordre d'origine
        result = np.empty(len(X))
        result[sidx] = preds_postproc
        return result


# ---------------------------------------------------------------------------
# Ensemble multi-graines
# ---------------------------------------------------------------------------


class SeedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Entraîne N instances de TrueOffEnvelopeRegressor avec des graines différentes
    et retourne la moyenne de leurs prédictions.

    Parameters
    ----------
    seeds : list[int]
        Liste des graines aléatoires à utiliser.
    base_params : dict
        Hyperparamètres passés à chaque TrueOffEnvelopeRegressor (sauf random_state).
    """

    def __init__(self, seeds: list | None = None, base_params: dict | None = None):
        self.seeds = seeds if seeds is not None else [0, 1, 2]
        self.base_params = base_params if base_params is not None else {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeedEnsembleRegressor":
        self.estimators_ = []
        for seed in self.seeds:
            est = TrueOffEnvelopeRegressor(random_state=seed, **self.base_params)
            est.fit(X, y)
            self.estimators_.append(est)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)
        return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Graphe skrub DataOps
# ---------------------------------------------------------------------------


def build_learner(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner pour l'expérience 02 (non entraîné)."""
    if data_preview is not None:
        data = skrub.var("data", value=data_preview)
    else:
        data = skrub.var("data")

    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data["patient_id"]},
    )
    y = data["target"].skb.mark_as_y()

    predictions = X.skb.apply(TrueOffEnvelopeRegressor(), y=y)
    return predictions.skb.make_learner()


def build_learner_03(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner pour l'expérience 03 : HGB accordé + ensemble 3 graines."""
    if data_preview is not None:
        data = skrub.var("data", value=data_preview)
    else:
        data = skrub.var("data")

    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data["patient_id"]},
    )
    y = data["target"].skb.mark_as_y()

    hgb_params = dict(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
    )
    estimator = SeedEnsembleRegressor(seeds=[0, 1, 2], base_params=hgb_params)
    predictions = X.skb.apply(estimator, y=y)
    return predictions.skb.make_learner()


# ---------------------------------------------------------------------------
# DebiasEnvelopeRegressor — enveloppe avec débiaisage fin (tbin, rater)
# ---------------------------------------------------------------------------


def _fit_debias_factors_full(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Apprend f = median(mesure/target | tbin, rater_id) et q95(off/target | rater_id)
    pour les blocs 2-3 améliorés. Appris dans fit uniquement.
    """
    joined = df.copy()
    joined["_y"] = y.values
    sub = joined[joined["_y"] > 3].copy()

    # OFF : facteur par rater (médiane du ratio off/target)
    has_off = sub["off"].notna()
    sub_off = sub[has_off].copy()
    sub_off["ratio"] = sub_off["off"] / sub_off["_y"]
    off_f = sub_off.groupby("rater_id")["ratio"].median().to_dict()
    off_q95 = sub_off.groupby("rater_id")["ratio"].quantile(0.95).to_dict()
    off_f_global = float(sub_off["ratio"].median())
    off_q95_global = float(sub_off["ratio"].quantile(0.95))

    # ON : facteur par (tbin, rater_id)
    has_on = sub["on"].notna()
    sub_on = sub[has_on].copy()
    tbin_on = _bin_on_time(sub_on["time_since_intake_on"])
    sub_on = sub_on.copy()
    sub_on["tbin"] = tbin_on.values
    sub_on["ratio"] = sub_on["on"] / sub_on["_y"]
    on_f = sub_on.groupby(["tbin", "rater_id"])["ratio"].median().to_dict()
    on_f_rater = sub_on.groupby("rater_id")["ratio"].median().to_dict()
    on_f_global = float(sub_on["ratio"].median())

    return {
        "off_f": off_f, "off_f_global": off_f_global,
        "off_q95": off_q95, "off_q95_global": off_q95_global,
        "on_f": on_f, "on_f_rater": on_f_rater, "on_f_global": on_f_global,
    }


def _apply_debias_corrections(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """
    Calcule off_c, on_eq, u_off, u_on, u_any avec le débiaisage fin.

    - off_c = off / f_rater   (estimation de la cible depuis off)
    - on_eq = on  / f_(tbin,rater)  (estimation de la cible depuis on)
    - u_off = off / q95_rater  (pour l'enveloppe — fraction du bord supérieur)
    - u_on  = on_eq / q95_rater
    - u_any = u_off si disponible, sinon u_on
    """
    out = df.copy()
    rater = df["rater_id"]
    tbin_on = _bin_on_time(df["time_since_intake_on"])

    f_off = rater.map(factors["off_f"]).fillna(factors["off_f_global"])
    q95 = rater.map(factors["off_q95"]).fillna(factors["off_q95_global"])

    out["off_c"] = np.where(df["off"].notna(), df["off"] / f_off.values, np.nan)

    # on_eq : facteur par (tbin, rater), fallback rater, fallback global
    on_f_vals = np.full(len(df), factors["on_f_global"])
    for i, (tb, rid) in enumerate(zip(tbin_on.values, rater.values)):
        key = (int(tb), rid)
        if key in factors["on_f"]:
            on_f_vals[i] = factors["on_f"][key]
        elif rid in factors["on_f_rater"]:
            on_f_vals[i] = factors["on_f_rater"][rid]
    out["on_eq"] = np.where(df["on"].notna(), df["on"] / on_f_vals, np.nan)

    out["u_off"] = np.where(df["off"].notna(), df["off"] / q95.values, np.nan)
    out["u_on"] = np.where(out["on_eq"].notna(), out["on_eq"] / q95.values, np.nan)
    out["u_any"] = np.where(out["u_off"].notna(), out["u_off"], out["u_on"])
    # Colonnes compatibles avec _fit_patient_line (attend off_c et on_c)
    out["on_c"] = out["on_eq"]
    return out


class DebiasEnvelopeRegressor(BaseEstimator, RegressorMixin):
    """
    Comme TrueOffEnvelopeRegressor mais avec un débiaisage fin par (tbin, rater_id)
    pour ON et par rater_id pour OFF, à la place des blocs 2-3 originaux.

    Hyperparamètres HGB identiques à ceux testés dans 03_hgb_tuned.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.5,
        random_state: int = 0,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _sorted_idx(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X.reset_index(drop=True)
            .sort_values(["patient_id", "age"], kind="stable")
            .index.to_numpy()
        )

    def _build_features(self, X_sorted: pd.DataFrame) -> pd.DataFrame:
        df = _block1_visit_features(X_sorted)
        df = _apply_debias_corrections(df, self.debias_factors_)
        # Estimations individuelles débiaisées comme features supplémentaires
        df_est = _build_estimates(df, self.factors_est_)
        df["est_off"] = df_est["est_off"].values
        df["est_on"] = df_est["est_on"].values
        df["est_any"] = df_est["est"].values
        df["w_off"] = df_est["w_off"].values
        df["w_on"] = df_est["w_on"].values
        df = _fit_patient_line(df)
        df = _patient_summary_features(df)
        df = _envelope_features(df)
        return df

    def _select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list, int | None]:
        env_cols = [c for c in df.columns if c.startswith("env_")]
        est_cols = ["est_off", "est_on", "est_any", "w_off", "w_on"]
        feature_cols = BASE_COLS + POOL_COLS + TREAT_COLS + est_cols + env_cols
        feature_cols = [c for c in feature_cols if c in df.columns]
        rater_idx = feature_cols.index("rater") if "rater" in feature_cols else None
        return df[feature_cols], feature_cols, rater_idx

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DebiasEnvelopeRegressor":
        sidx = self._sorted_idx(X)
        X_sorted = X.iloc[sidx].reset_index(drop=True)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        y_sorted = pd.Series(y_arr[sidx], name="target")

        df = _block1_visit_features(X_sorted)

        # Facteurs de débiaisage (remplacent blocs 2-3)
        self.debias_factors_ = _fit_debias_factors_full(df, y_sorted)
        # Facteurs pour les estimations individuelles (utilisés dans _build_estimates)
        self.factors_est_ = _fit_debias_factors(df, y_sorted)
        df = _apply_debias_corrections(df, self.debias_factors_)

        # Estimations individuelles comme features
        df_est = _build_estimates(df, self.factors_est_)
        df["est_off"] = df_est["est_off"].values
        df["est_on"] = df_est["est_on"].values
        df["est_any"] = df_est["est"].values
        df["w_off"] = df_est["w_off"].values
        df["w_on"] = df_est["w_on"].values

        df = _fit_patient_line(df)
        df = _patient_summary_features(df)
        df = _envelope_features(df)

        feat_df, self.feature_cols_, rater_idx = self._select_features(df)
        cat_features = [rater_idx] if rater_idx is not None else []

        self.booster_ = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            categorical_features=cat_features if cat_features else None,
            random_state=self.random_state,
        )
        self.booster_.fit(feat_df.values, y_sorted.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        sidx = self._sorted_idx(X)
        X_sorted = X.iloc[sidx].reset_index(drop=True)
        df = self._build_features(X_sorted)
        feat_df = df[self.feature_cols_]
        raw_preds = self.booster_.predict(feat_df.values)
        preds_postproc = _postprocess_predictions(raw_preds, df)
        result = np.empty(len(X))
        result[sidx] = preds_postproc
        return result


def build_learner_debias_envelope(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner pour l'expérience 03_debias_wls (DebiasEnvelopeRegressor)."""
    if data_preview is not None:
        data = skrub.var("data", value=data_preview)
    else:
        data = skrub.var("data")

    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data["patient_id"]},
    )
    y = data["target"].skb.mark_as_y()

    predictions = X.skb.apply(DebiasEnvelopeRegressor(), y=y)
    return predictions.skb.make_learner()


# ---------------------------------------------------------------------------
# DebiasWLSRegressor — étage mesure + WLS parabolique par patient
# ---------------------------------------------------------------------------


# Tranches de délai ON pour le débiaisage (heures depuis prise)
_ON_TIME_BINS = [-0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]


def _bin_on_time(s: pd.Series) -> pd.Series:
    """Discrétise time_since_intake_on en indices de tranche (sentinelle -1 si NaN)."""
    binned = pd.cut(s.fillna(-99.0), bins=_ON_TIME_BINS, labels=False, right=True, include_lowest=True)
    return binned.fillna(-1).astype(int)


def _fit_debias_factors(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Apprend les facteurs de débiaisage f = mediane(mesure/target | tbin, rater_id)
    et la variance relative var_rel = std(mesure/target | tbin, rater_id)^2
    séparément pour ON et OFF.

    Fallbacks : (rater_id,) puis global.
    Retourne dict avec clés 'off_f', 'on_f', 'off_var', 'on_var' chacun
    étant un dict (tbin, rater_id) -> float.
    """
    joined = df.copy()
    joined["_y"] = y.values
    joined = joined[joined["_y"] > 3].copy()

    result = {}
    for meas in ("off", "on"):
        has = joined[meas].notna()
        sub = joined[has].copy()
        sub["ratio"] = sub[meas] / sub["_y"]

        if meas == "on":
            sub["tbin"] = _bin_on_time(sub["time_since_intake_on"])
            key_cols = ["tbin", "rater_id"]
        else:
            key_cols = ["rater_id"]

        f_map = sub.groupby(key_cols)["ratio"].median().to_dict()
        var_map = (sub.groupby(key_cols)["ratio"].std() ** 2).to_dict()

        # fallbacks
        f_rater = sub.groupby("rater_id")["ratio"].median().to_dict()
        var_rater = (sub.groupby("rater_id")["ratio"].std() ** 2).to_dict()
        f_global = float(sub["ratio"].median())
        var_global = float(sub["ratio"].std() ** 2)

        result[f"{meas}_f"] = f_map
        result[f"{meas}_var"] = var_map
        result[f"{meas}_f_rater"] = f_rater
        result[f"{meas}_var_rater"] = var_rater
        result[f"{meas}_f_global"] = f_global
        result[f"{meas}_var_global"] = max(var_global, 1e-6)

    return result


def _debias_single(val: float, f: float, var: float) -> tuple[float, float]:
    """Retourne (estimation cible, variance de l'estimation)."""
    if f < 1e-6:
        return np.nan, np.inf
    return val / f, max(var / (f ** 2), 1e-4)


def _get_factor(key, f_map, f_fallback_map, f_global, rater_id):
    """Récupère f et var avec fallbacks successifs."""
    if key in f_map:
        return f_map[key]
    if rater_id in f_fallback_map:
        return f_fallback_map[rater_id]
    return f_global


def _get_var(key, var_map, var_fallback_map, var_global, rater_id):
    if key in var_map:
        return max(var_map[key], 1e-6)
    if rater_id in var_fallback_map:
        return max(var_fallback_map[rater_id], 1e-6)
    return var_global


def _build_estimates(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """
    Calcule vectoriellement est_off, w_off, est_on, w_on, est, w pour chaque ligne.

    Pour OFF : clé = rater_id.
    Pour ON  : clé = (tbin, rater_id).
    Fallbacks : rater seul, puis global.
    """
    tbin_on = _bin_on_time(df["time_since_intake_on"]).values
    rater = df["rater_id"].values
    off_vals = df["off"].values.astype(float)
    on_vals = df["on"].values.astype(float)

    n = len(df)
    est_off = np.full(n, np.nan)
    w_off = np.zeros(n)
    est_on = np.full(n, np.nan)
    w_on = np.zeros(n)

    # --- OFF (key = rater_id) ---
    has_off = np.isfinite(off_vals) & (off_vals > 0)
    for i in np.where(has_off)[0]:
        rid = rater[i]
        f = factors["off_f"].get(rid, factors["off_f_rater"].get(rid, factors["off_f_global"]))
        v = factors["off_var"].get(rid, factors["off_var_rater"].get(rid, factors["off_var_global"]))
        if f > 1e-6:
            est_off[i] = off_vals[i] / f
            w_off[i] = f ** 2 / max(v, 1e-6)

    # --- ON (key = (tbin, rater_id)) ---
    has_on = np.isfinite(on_vals) & (on_vals > 0)
    for i in np.where(has_on)[0]:
        rid = rater[i]; tb = int(tbin_on[i])
        key = (tb, rid)
        f = factors["on_f"].get(key, factors["on_f_rater"].get(rid, factors["on_f_global"]))
        v = factors["on_var"].get(key, factors["on_var_rater"].get(rid, factors["on_var_global"]))
        if f > 1e-6:
            est_on[i] = on_vals[i] / f
            w_on[i] = f ** 2 / max(v, 1e-6)

    both = np.isfinite(est_off) & np.isfinite(est_on)
    only_off = np.isfinite(est_off) & ~np.isfinite(est_on)
    only_on = ~np.isfinite(est_off) & np.isfinite(est_on)

    out = df.copy()
    out["est_off"] = est_off
    out["w_off"] = w_off
    out["est_on"] = est_on
    out["w_on"] = w_on
    out["est"] = np.where(
        both, (w_off * est_off + w_on * est_on) / (w_off + w_on),
        np.where(only_off, est_off, np.where(only_on, est_on, np.nan))
    )
    out["w"] = np.where(both, w_off + w_on, np.where(only_off, w_off, np.where(only_on, w_on, 0.0)))
    return out


def _fit_population_prior(df_with_est: pd.DataFrame, y: pd.Series) -> dict:
    """
    Ajuste une parabole par patient sur la CIBLE (disponible en train) et
    extrait la distribution (moyenne, std) des 3 coefficients en population.
    Utilisée comme a priori de shrinkage.
    """
    df = df_with_est.copy()
    df["_y"] = y.values
    c0s, c1s, c2s = [], [], []
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("age")
        t = grp["_y"].values
        a = grp["age"].values - grp["age"].mean()
        if len(grp) >= 3:
            c2, c1, c0 = np.polyfit(a, t, 2)
        elif len(grp) == 2:
            c1, c0 = np.polyfit(a, t, 1); c2 = 0.0
        else:
            c0 = float(t[0]); c1 = 0.0; c2 = 0.0
        c0s.append(c0); c1s.append(c1); c2s.append(c2)
    return {
        "c0_mean": float(np.mean(c0s)), "c0_std": max(float(np.std(c0s)), 1.0),
        "c1_mean": float(np.mean(c1s)), "c1_std": max(float(np.std(c1s)), 0.1),
        "c2_mean": float(np.mean(c2s)), "c2_std": max(float(np.std(c2s)), 0.01),
    }


def _quantile_parabola_per_patient(df: pd.DataFrame, prior: dict, quantile: float = 0.90) -> np.ndarray:
    """
    Pour chaque patient, ajuste une parabole en âge centré par régression quantile
    (quantile 90%) sur ses estimations débiaisées, avec shrinkage empirical Bayes.

    Le quantile élevé exploite le bruit à sens unique : les mesures hautes bornent
    la courbe réelle par le bas, donc le quantile 90% estime mieux que la médiane.

    Retourne un tableau de prédictions (même ordre que df).
    """
    from scipy.optimize import minimize

    preds = np.full(len(df), np.nan)

    lam_c0 = 1.0 / prior["c0_std"] ** 2
    lam_c1 = 1.0 / prior["c1_std"] ** 2
    lam_c2 = 1.0 / prior["c2_std"] ** 2
    prior_vec = np.array([prior["c0_mean"], prior["c1_mean"], prior["c2_mean"]])

    def _pinball(params, ac, ev, lam0, lam1, lam2, pv, q):
        c0, c1, c2 = params
        fitted = c0 + c1 * ac + c2 * ac ** 2
        resid = ev - fitted
        loss = np.where(resid >= 0, q * resid, (q - 1) * resid).mean()
        reg = 0.5 * (lam0 * (c0 - pv[0]) ** 2 + lam1 * (c1 - pv[1]) ** 2 + lam2 * (c2 - pv[2]) ** 2)
        return loss + reg

    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("age")
        idx = grp.index.to_numpy()
        age = grp["age"].values
        a0 = age.mean()
        ac = age - a0

        est = grp["est"].values
        valid = np.isfinite(est)
        n_valid = int(valid.sum())

        if n_valid == 0:
            preds[idx] = np.clip(prior["c0_mean"] + prior["c1_mean"] * ac, 0, 132)
            continue

        ev = est[valid]; acv = ac[valid]

        if n_valid >= 3:
            res = minimize(
                _pinball, x0=prior_vec.copy(), args=(acv, ev, lam_c0, lam_c1, lam_c2, prior_vec, quantile),
                method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 500},
            )
            c0, c1, c2 = res.x
        elif n_valid == 2:
            # OLS shrunk pour 2 points (quantile instable)
            A = np.column_stack([np.ones(2), acv])
            P = np.diag([lam_c0, lam_c1])
            pv2 = prior_vec[:2]
            coeffs = np.linalg.solve(A.T @ A + P, A.T @ ev + P @ pv2)
            c0, c1 = coeffs; c2 = prior["c2_mean"]
        else:
            c0 = (ev[0] * lam_c0 * 5 + prior["c0_mean"] * lam_c0) / (lam_c0 * 6)
            c1 = prior["c1_mean"]; c2 = prior["c2_mean"]

        p = c0 + c1 * ac + c2 * ac ** 2
        if len(p) > 1:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            p = iso.fit_transform(age, p)
        preds[idx] = np.clip(p, 0.0, 132.0)

    return preds


class DebiasWLSRegressor(BaseEstimator, RegressorMixin):
    """
    Estimateur scikit-learn : débiaisage des mesures ON/OFF par facteurs appris,
    puis régression quantile (q=0.90) d'une parabole par patient avec shrinkage
    empirical Bayes. Exploite le bruit à sens unique : les mesures hautes bornent
    mieux la courbe vraie que la moyenne.

    Tout ce qui utilise la cible est appris dans fit.
    patient_id et age doivent être présents dans X.
    """

    def __init__(self, quantile: float = 0.90):
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DebiasWLSRegressor":
        X = X.reset_index(drop=True)
        y = pd.Series(y.values if hasattr(y, "values") else np.asarray(y), name="target")

        order = X.sort_values(["patient_id", "age"]).index.to_numpy()
        Xs = X.iloc[order].reset_index(drop=True)
        ys = y.iloc[order].reset_index(drop=True)

        self.factors_ = _fit_debias_factors(Xs, ys)
        df_est = _build_estimates(Xs, self.factors_)
        self.prior_ = _fit_population_prior(df_est, ys)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        order = X.sort_values(["patient_id", "age"]).index.to_numpy()
        Xs = X.iloc[order].reset_index(drop=True)

        df_est = _build_estimates(Xs, self.factors_)
        preds_sorted = _quantile_parabola_per_patient(df_est, self.prior_, self.quantile)

        result = np.empty(len(X))
        result[order] = preds_sorted
        return result


def build_learner_debias_wls(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner pour l'expérience 03_debias_wls."""
    if data_preview is not None:
        data = skrub.var("data", value=data_preview)
    else:
        data = skrub.var("data")

    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data["patient_id"]},
    )
    y = data["target"].skb.mark_as_y()

    predictions = X.skb.apply(DebiasWLSRegressor(), y=y)
    return predictions.skb.make_learner()


# ---------------------------------------------------------------------------
# StackingEnvelopeRegressor — HGB accordé + parabole quantile, combinés
# ---------------------------------------------------------------------------


class StackingEnvelopeRegressor(BaseEstimator, RegressorMixin):
    """
    Combine TrueOffEnvelopeRegressor (HGB accordé) et DebiasWLSRegressor
    (parabole quantile par patient) via un stacking avec CV interne GroupKFold(3).

    La prédiction finale est une combinaison linéaire apprise sur les prédictions
    out-of-fold en train : alpha * HGB + (1-alpha) * WLS.

    Tout ce qui utilise la cible est appris dans fit.
    """

    def __init__(
        self,
        hgb_params: dict | None = None,
        wls_quantile: float = 0.90,
        random_state: int = 0,
    ):
        self.hgb_params = hgb_params
        self.wls_quantile = wls_quantile
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnvelopeRegressor":
        X = X.reset_index(drop=True)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        hgb_p = self.hgb_params or {
            "learning_rate": 0.05, "max_iter": 1000, "max_leaf_nodes": 63,
            "min_samples_leaf": 20, "l2_regularization": 0.5,
        }

        # --- Out-of-fold predictions pour apprendre alpha ---
        oof_hgb = np.zeros(len(X))
        oof_wls = np.zeros(len(X))
        gkf3 = GroupKFold(n_splits=3)
        for tr_i, val_i in gkf3.split(X, y_arr, groups=X["patient_id"]):
            X_tr = X.iloc[tr_i].reset_index(drop=True)
            y_tr = pd.Series(y_arr[tr_i])
            X_val = X.iloc[val_i].reset_index(drop=True)

            hgb_fold = TrueOffEnvelopeRegressor(random_state=self.random_state, **hgb_p)
            hgb_fold.fit(X_tr, y_tr)
            oof_hgb[val_i] = hgb_fold.predict(X_val)

            wls_fold = DebiasWLSRegressor(quantile=self.wls_quantile)
            wls_fold.fit(X_tr, y_tr)
            oof_wls[val_i] = wls_fold.predict(X_val)

        # alpha par régression ridge sur les OOF
        A = np.column_stack([oof_hgb, oof_wls])
        # Ridge avec lambda=10 pour éviter le surapprentissage
        self.alpha_ = np.linalg.solve(A.T @ A + 10 * np.eye(2), A.T @ y_arr)

        # --- Modèles finaux entraînés sur tout le train ---
        self.hgb_final_ = TrueOffEnvelopeRegressor(random_state=self.random_state, **hgb_p)
        self.hgb_final_.fit(X, pd.Series(y_arr))

        self.wls_final_ = DebiasWLSRegressor(quantile=self.wls_quantile)
        self.wls_final_.fit(X, pd.Series(y_arr))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_hgb = self.hgb_final_.predict(X)
        p_wls = self.wls_final_.predict(X)
        A = np.column_stack([p_hgb, p_wls])
        preds = A @ self.alpha_
        return np.clip(preds, 0.0, 132.0)


def build_learner_stacking(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner pour l'expérience 03_debias_wls (stacking)."""
    if data_preview is not None:
        data = skrub.var("data", value=data_preview)
    else:
        data = skrub.var("data")

    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data["patient_id"]},
    )
    y = data["target"].skb.mark_as_y()

    predictions = X.skb.apply(StackingEnvelopeRegressor(), y=y)
    return predictions.skb.make_learner()
