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
# Graphe skrub DataOps
# ---------------------------------------------------------------------------


def build_learner(data_preview: pd.DataFrame | None = None) -> object:
    """Construit le SkrubLearner (non entraîné)."""
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
