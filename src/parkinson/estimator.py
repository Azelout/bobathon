"""
Estimateur True OFF à enveloppe pour la prédiction du score MDS-UPDRS III OFF.

Implémente les six blocs de la stratégie décrite dans journal/STRATEGIE.md :
1. Variables de visite et de traitement (sans cible)
2. Conversion de `on` en équivalent `off` (sans cible, courbe apprise dans fit)
3. Corrections par évaluateur (apprises dans fit)
4. Variables par patient : droite pondérée + résumés + enveloppe 90 colonnes
5. HistGradientBoostingRegressor
6. Post-traitement par patient : parabole → isotonique → clip [0, 132]

Options supplémentaires (expérience 03) :
- sens_neighbors : a priori bayésien de sensibilité emprunté aux voisins en âge au diagnostic.
- use_posterior : OneSidedPosterior empilé hors pli (post_mean, post_sd).
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


def _fit_sens_reference(df: pd.DataFrame, curve: dict, lam: float = 1.0) -> pd.DataFrame:
    """
    Construit la table de référence des patients avec au moins une visite double.

    Colonnes de sortie : patient_id, age_at_diagnosis, r_j (résidu moyen), w_j (poids).
    """
    doubles = df[
        df["off"].notna() & df["on"].notna() & (df["off"] >= 8) & (df["on"] >= 1)
    ].copy()
    doubles["log_ratio"] = np.log(doubles["on"] / doubles["off"])
    doubles["curve_val"] = doubles["hb"].map(curve).fillna(0.0)
    doubles["residual"] = doubles["log_ratio"] - doubles["curve_val"]

    agg = doubles.groupby("patient_id")["residual"].agg(["sum", "count"])
    agg = agg[agg["count"] >= 1].copy()
    agg["r_j"] = agg["sum"] / agg["count"]
    agg["w_j"] = agg["count"] / (agg["count"] + lam)

    # Récupérer age_at_diagnosis (première valeur non nulle par patient)
    age_diag = (
        df[df["age_at_diagnosis"].notna()]
        .groupby("patient_id")["age_at_diagnosis"]
        .first()
    )
    ref = agg.join(age_diag, how="left")
    return ref[["r_j", "w_j", "age_at_diagnosis"]].reset_index()


def _compute_sens_prior(
    patient_ages: np.ndarray,
    patient_ids: np.ndarray,
    ref: pd.DataFrame,
    sigma: float = 0.05,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Calcule l'a priori de sensibilité pour chaque ligne.

    Parameters
    ----------
    patient_ages : 1-D array of float, age_at_diagnosis par ligne (NaN si manquant).
    patient_ids : 1-D array, patient_id par ligne.
    ref : DataFrame de la table de référence (patient_id, age_at_diagnosis, r_j, w_j).
    sigma : écart-type du noyau gaussien (en années).
    beta : pseudo-count d'a priori.
    """
    prior = np.zeros(len(patient_ages))

    # Construire un mapping patient_id → age_at_diagnosis (représentant unique par patient)
    # et patient_id → sum_residuals, count pour éviter l'auto-influence
    ref_pid = ref["patient_id"].values
    ref_age = ref["age_at_diagnosis"].values
    ref_r = ref["r_j"].values
    ref_w = ref["w_j"].values

    # Index de la table de référence par patient_id (pour l'exclusion du patient lui-même)
    ref_pid_set = {pid: i for i, pid in enumerate(ref_pid)}

    for row_i in range(len(patient_ages)):
        a = patient_ages[row_i]
        pid = patient_ids[row_i]
        if np.isnan(a):
            prior[row_i] = 0.0
            continue

        # Calculer les poids du noyau pour tous les patients de référence
        diff = ref_age - a
        valid = ~np.isnan(diff)
        k = np.where(valid, np.exp(-0.5 * (diff / sigma) ** 2), 0.0)

        # Exclure le patient lui-même
        if pid in ref_pid_set:
            k[ref_pid_set[pid]] = 0.0

        wk = ref_w * k
        num = np.sum(wk * ref_r)
        denom = np.sum(wk) + beta
        prior[row_i] = num / denom if denom > 0 else 0.0

    return prior


def _add_on_equivalent(
    df: pd.DataFrame,
    curve: dict,
    lam: float = 1.0,
    sens_neighbors: bool = False,
    sens_ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcule la sensibilité par patient (shrunk mean residual) et on_eq = on * exp(-(courbe+sens)).

    Si sens_neighbors=True, la sensibilité utilise un a priori bayésien emprunté aux
    patients d'entraînement voisins en age_at_diagnosis (table sens_ref fournie en fit).
    En predict, sens_ref provient toujours du train seul → isolation préservée.
    """
    doubles = df[
        df["off"].notna() & df["on"].notna() & (df["off"] >= 8) & (df["on"] >= 1)
    ].copy()
    doubles["log_ratio"] = np.log(doubles["on"] / doubles["off"])
    doubles["curve_val"] = doubles["hb"].map(curve).fillna(0.0)
    doubles["residual"] = doubles["log_ratio"] - doubles["curve_val"]

    agg = doubles.groupby("patient_id")["residual"].agg(["sum", "count"])
    nboth_map = agg["count"].to_dict()

    out = df.copy()
    out["nboth"] = df["patient_id"].map(nboth_map).fillna(0).astype(int)

    if sens_neighbors and sens_ref is not None:
        # A priori bayésien
        patient_ages = df["age_at_diagnosis"].values.astype(float)
        patient_ids = df["patient_id"].values
        prior_vals = _compute_sens_prior(patient_ages, patient_ids, sens_ref)

        # Sensibilité = (sum + 2*prior) / (count + 2)
        sum_map = agg["sum"].to_dict()
        count_map = agg["count"].to_dict()

        sens_vals = np.zeros(len(df))
        for row_i, pid in enumerate(df["patient_id"].values):
            s = sum_map.get(pid, 0.0)
            c = count_map.get(pid, 0)
            sens_vals[row_i] = (s + 2.0 * prior_vals[row_i]) / (c + 2.0)

        out["sens"] = sens_vals
        out["sens_prior"] = prior_vals
    else:
        sens_map = (agg["sum"] / (agg["count"] + lam)).to_dict()
        out["sens"] = df["patient_id"].map(sens_map).fillna(0.0)
        if sens_neighbors:
            # sens_neighbors=True mais pas de référence (cas de la table vide)
            out["sens_prior"] = 0.0

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
# Bloc 7 — Modèle bayésien OneSidedPosterior
# ---------------------------------------------------------------------------

# Grilles
_L_GRID = np.arange(0.0, 130.5, 0.5)   # 261 valeurs
_S_GRID = np.arange(0.0, 12.25, 0.25)  # 49 valeurs
_C_GRID = np.arange(-0.3, 0.31, 0.1)   # 7 valeurs

# Tranches de cible pour la densité de bruit
_T_EDGES = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, np.inf]

# Tranches de T pour la vraisemblance de début de traitement
_TREAT_T_EDGES = [0.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 40.0, np.inf]
_TREAT_DUR_EDGES = [-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf]


def _smooth_density(counts: np.ndarray, sigma_bins: float = 2.5, floor: float = 0.02) -> np.ndarray:
    """Lisse un histogramme par noyau gaussien discret (±3σ tronqué)."""
    half = int(np.ceil(3 * sigma_bins))
    kernel_x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts.astype(float), kernel, mode="same")
    smoothed += floor
    # Normaliser en densité (intégrale = 1 sur l'intervalle [0, 2], pas = 0.01)
    smoothed /= smoothed.sum() * 0.01
    return smoothed


def _z_density(z_vals: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """
    Estime la densité empirique sur z ∈ [0, 2] par pas de 0.01 (201 bins).
    Retourne un array de 201 valeurs de densité.
    """
    z_clipped = np.clip(z_vals, 0.0, 2.0)
    counts, _ = np.histogram(z_clipped, bins=201, range=(0.0, 2.0))
    return _smooth_density(counts)


def _t_bin(T_val: float) -> int:
    """Index de tranche de cible pour la densité de bruit."""
    for i, edge in enumerate(_T_EDGES[1:]):
        if T_val < edge:
            return i
    return len(_T_EDGES) - 2


def _assign_bin_scalar(val: float, edges: list) -> int:
    """Cherche le bin d'une valeur scalaire dans une liste de bords."""
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return i
    return len(edges) - 2


class OneSidedPosterior(BaseEstimator):
    """
    Modèle bayésien par patient sur une grille L×S×C.

    Utilise les colonnes : treat, dur, on_eq, nboth, off, rater_id,
    time_since_intake_on (déjà calculées par les blocs 1-2).

    Parameters
    ----------
    None — tous les paramètres sont appris dans fit.
    """

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "OneSidedPosterior":
        """
        Apprend les densités de bruit, la table de traitement et les a priori.

        Parameters
        ----------
        df : DataFrame avec les colonnes blocs 1-2 + rater_id + age_at_diagnosis.
        y : cible (true OFF score).
        """
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        # ------------------------------------------------------------------
        # Échelle par évaluateur : médiane de off/cible sur visites avec off et cible > 20
        # ------------------------------------------------------------------
        df2 = df.copy()
        df2["target"] = y_arr
        valid_scale = df2[df2["off"].notna() & (df2["target"] > 20)].copy()
        valid_scale["ratio"] = valid_scale["off"] / valid_scale["target"]
        self.rater_scale_ = valid_scale.groupby("rater_id")["ratio"].median().to_dict()
        self.global_scale_ = float(valid_scale["ratio"].median()) if len(valid_scale) > 0 else 1.0

        # ------------------------------------------------------------------
        # Densités empiriques de bruit pour off
        # ------------------------------------------------------------------
        # Une densité par tranche de T et par treat (0/1)
        n_tbins = len(_T_EDGES) - 1
        # off_densities[treat_val][t_bin] = array 201
        self.off_densities_: dict[int, list] = {0: [], 1: []}
        for treat_val in [0, 1]:
            for t_bin_i in range(n_tbins):
                sub = df2[
                    df2["off"].notna()
                    & (df2["treat"] == treat_val)
                    & (df2["target"] > 0)
                ].copy()
                # filtre tranche de T (target utilisée comme proxy de T ici)
                t_lo = _T_EDGES[t_bin_i]
                t_hi = _T_EDGES[t_bin_i + 1]
                sub_t = sub[(sub["target"] >= t_lo) & (sub["target"] < t_hi)]
                if len(sub_t) < 5:
                    sub_t = sub  # fallback global
                scale_vals = sub_t["rater_id"].map(self.rater_scale_).fillna(self.global_scale_)
                T_proxy = sub_t["target"].values
                T_clamped = np.maximum(T_proxy, 0.5)
                z = sub_t["off"].values / (scale_vals.values * T_clamped)
                self.off_densities_[treat_val].append(_z_density(z))

        # ------------------------------------------------------------------
        # Densités empiriques de bruit pour on_eq
        # Une densité par combinaison (time_known: bool) × (nboth > 0: bool)
        # ------------------------------------------------------------------
        self.on_densities_: dict[tuple, list] = {}
        for time_known in [False, True]:
            for nboth_pos in [False, True]:
                key = (time_known, nboth_pos)
                self.on_densities_[key] = []
                for t_bin_i in range(n_tbins):
                    t_lo = _T_EDGES[t_bin_i]
                    t_hi = _T_EDGES[t_bin_i + 1]
                    sub = df2[
                        df2["off"].isna()
                        & df2["on_eq"].notna()
                        & (df2["target"] > 0)
                    ].copy()
                    if time_known:
                        sub = sub[sub["time_since_intake_on"].notna()]
                    else:
                        sub = sub[sub["time_since_intake_on"].isna()]
                    if nboth_pos:
                        sub = sub[sub["nboth"] > 0]
                    else:
                        sub = sub[sub["nboth"] == 0]
                    sub_t = sub[(sub["target"] >= t_lo) & (sub["target"] < t_hi)]
                    if len(sub_t) < 5:
                        sub_t = sub
                    if len(sub_t) == 0:
                        self.on_densities_[key].append(np.ones(201) / (201 * 0.01))
                        continue
                    scale_vals = sub_t["rater_id"].map(self.rater_scale_).fillna(self.global_scale_)
                    T_proxy = sub_t["target"].values
                    T_clamped = np.maximum(T_proxy, 0.5)
                    z = sub_t["on_eq"].values / (scale_vals.values * T_clamped)
                    self.on_densities_[key].append(_z_density(z))

        # ------------------------------------------------------------------
        # Table de probabilité de début de traitement
        # ------------------------------------------------------------------
        n_treat_t = len(_TREAT_T_EDGES) - 1
        n_treat_dur = len(_TREAT_DUR_EDGES) - 1

        # Détecter les visites de "début de traitement" :
        # treat=1 et visite précédente du même patient est treat=0 (ou première visite)
        df_sorted = df2.sort_values(["patient_id", "age"])
        df_sorted["prev_treat"] = df_sorted.groupby("patient_id")["treat"].shift(1).fillna(0)
        onset = df_sorted[
            (df_sorted["treat"] == 1) & (df_sorted["prev_treat"] == 0)
        ].copy()

        # Médiane de dur pour les dur manquantes
        self.dur_median_ = float(df2["dur"].median()) if df2["dur"].notna().any() else 0.0

        # Table de succès / effectif
        self.treat_start_table_ = np.zeros((n_treat_t, n_treat_dur, 2))  # [:, :, 0]=succès [:, :, 1]=effectif

        # Pour chaque visite où treat=0, compter les occasions
        non_treat = df_sorted[df_sorted["treat"] == 0].copy()
        non_treat_dur = non_treat["dur"].fillna(self.dur_median_).values
        non_treat_T = non_treat["target"].values  # proxy T
        for i in range(len(non_treat)):
            tb = _assign_bin_scalar(float(non_treat_T[i]), _TREAT_T_EDGES)
            db = _assign_bin_scalar(float(non_treat_dur[i]), _TREAT_DUR_EDGES)
            if 0 <= tb < n_treat_t and 0 <= db < n_treat_dur:
                self.treat_start_table_[tb, db, 1] += 1

        onset_dur = onset["dur"].fillna(self.dur_median_).values
        onset_T = onset["target"].values
        for i in range(len(onset)):
            tb = _assign_bin_scalar(float(onset_T[i]), _TREAT_T_EDGES)
            db = _assign_bin_scalar(float(onset_dur[i]), _TREAT_DUR_EDGES)
            if 0 <= tb < n_treat_t and 0 <= db < n_treat_dur:
                self.treat_start_table_[tb, db, 0] += 1

        # Probabilités de Laplace
        denom = self.treat_start_table_[:, :, 1] + 1.0
        self.treat_start_prob_ = np.clip(
            (self.treat_start_table_[:, :, 0] + 0.5) / denom,
            0.003, 0.997
        )

        # ------------------------------------------------------------------
        # A priori niveau L : régression linéaire sur [1, dur, dur², part traitée]
        # ------------------------------------------------------------------
        pat_stats = []
        for pid, grp in df2.groupby("patient_id"):
            t_mean = float(grp["target"].mean())
            dur_mean = float(grp["dur"].mean()) if grp["dur"].notna().any() else self.dur_median_
            dur2 = dur_mean ** 2
            treat_frac = float(grp["treat"].mean())
            pat_stats.append([1.0, dur_mean, dur2, treat_frac, t_mean])

        if len(pat_stats) > 4:
            pat_arr = np.array(pat_stats)
            X_reg = pat_arr[:, :4]
            y_reg = pat_arr[:, 4]
            # OLS avec régularisation
            XtX = X_reg.T @ X_reg + 1e-4 * np.eye(4)
            Xty = X_reg.T @ y_reg
            self.prior_L_coef_ = np.linalg.solve(XtX, Xty)
            resid = y_reg - X_reg @ self.prior_L_coef_
            self.prior_L_sigma_ = float(np.std(resid) * 1.5)
        else:
            self.prior_L_coef_ = np.array([30.0, 0.0, 0.0, 0.0])
            self.prior_L_sigma_ = 15.0

        self.prior_L_sigma_ = max(self.prior_L_sigma_, 1.0)

        # ------------------------------------------------------------------
        # A priori pente S : histogramme des pentes linéaires
        # ------------------------------------------------------------------
        slopes_list = []
        for pid, grp in df2.groupby("patient_id"):
            targets = grp["target"].values
            ages = grp["age"].values
            if len(grp) < 2:
                continue
            age_c = ages - ages.mean()
            try:
                slope = float(np.polyfit(age_c, targets, 1)[0])
            except Exception:
                continue
            if np.isfinite(slope):
                slopes_list.append(slope)

        if slopes_list:
            slopes_arr = np.array(slopes_list)
            counts_s, _ = np.histogram(slopes_arr, bins=len(_S_GRID), range=(_S_GRID[0], _S_GRID[-1]))
            counts_s = counts_s.astype(float) + 1.0
            self.prior_S_ = counts_s / counts_s.sum()
        else:
            self.prior_S_ = np.ones(len(_S_GRID)) / len(_S_GRID)

        # ------------------------------------------------------------------
        # A priori courbure C : histogramme des coefficients quadratiques
        # ------------------------------------------------------------------
        curv_list = []
        for pid, grp in df2.groupby("patient_id"):
            if len(grp) < 5:
                continue
            targets = grp["target"].values
            ages = grp["age"].values
            age_c = ages - ages.mean()
            try:
                coeffs = np.polyfit(age_c, targets, 2)
                c_val = float(coeffs[0])
            except Exception:
                continue
            if np.isfinite(c_val):
                c_clipped = np.clip(c_val, _C_GRID[0], _C_GRID[-1])
                curv_list.append(c_clipped)

        if curv_list:
            curv_arr = np.array(curv_list)
            counts_c, _ = np.histogram(curv_arr, bins=len(_C_GRID), range=(_C_GRID[0], _C_GRID[-1]))
            counts_c = counts_c.astype(float) + 1.0
            self.prior_C_ = counts_c / counts_c.sum()
        else:
            self.prior_C_ = np.ones(len(_C_GRID)) / len(_C_GRID)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcule post_mean et post_sd pour chaque ligne.

        Retourne un array de shape (n, 2) : [:, 0] = post_mean, [:, 1] = post_sd.
        """
        # Réinitialiser l'index pour garantir des index positionnels 0..n-1
        df = df.reset_index(drop=True)
        n = len(df)
        post_mean = np.full(n, np.nan)
        post_sd = np.full(n, 10.0)

        for pid, grp in df.groupby("patient_id"):
            idx = grp.index.to_numpy()
            ages = grp["age"].values
            age_mean = ages.mean()
            x_pat = ages - age_mean  # x = âge − âge moyen

            # A priori sur L
            dur_m = float(grp["dur"].mean()) if grp["dur"].notna().any() else self.dur_median_
            treat_frac = float(grp["treat"].mean())
            reg_feat = np.array([1.0, dur_m, dur_m ** 2, treat_frac])
            prior_L_mean = float(reg_feat @ self.prior_L_coef_)
            prior_L_sigma = self.prior_L_sigma_

            nL = len(_L_GRID)
            nS = len(_S_GRID)
            nC = len(_C_GRID)

            # Log-a priori : shape (nL, nS, nC)
            # L : gaussien
            log_prior_L = -0.5 * ((_L_GRID - prior_L_mean) / prior_L_sigma) ** 2
            log_prior_S = np.log(np.maximum(self.prior_S_, 1e-30))  # shape (nS,)
            log_prior_C = np.log(np.maximum(self.prior_C_, 1e-30))  # shape (nC,)

            # log_posterior cumulé sur les visites du patient : shape (nL, nS, nC)
            log_post = (
                log_prior_L[:, None, None]
                + log_prior_S[None, :, None]
                + log_prior_C[None, None, :]
            )

            # Récupérer l'échelle de l'évaluateur du patient
            rater_id = grp["rater_id"].iloc[0] if "rater_id" in grp.columns else None
            scale = self.rater_scale_.get(rater_id, self.global_scale_) if rater_id is not None else self.global_scale_

            # Calcul de x² moyen du patient
            x2_mean = float(np.mean(x_pat ** 2))

            for visit_i, (vi, row) in enumerate(grp.iterrows()):
                # T(x) = L + S*x + C*(x² − x²_mean)  →  shape (nL, nS, nC)
                x_v = float(x_pat[visit_i])
                x2_v = float(x_v ** 2 - x2_mean)

                # T_grid[l, s, c] = L_grid[l] + S_grid[s]*x_v + C_grid[c]*x2_v
                T_grid = (
                    _L_GRID[:, None, None]
                    + _S_GRID[None, :, None] * x_v
                    + _C_GRID[None, None, :] * x2_v
                )
                T_grid_clamped = np.maximum(T_grid, 0.5)

                # Tranche de T (pour la densité)
                T_repr = np.median(T_grid)  # représentant approximatif
                t_bin_i = _t_bin(float(T_repr))

                treat_val = int(row["treat"]) if not pd.isna(row["treat"]) else 0

                # Vraisemblance off
                if pd.notna(row.get("off")):
                    z_off = float(row["off"]) / (scale * T_grid_clamped)
                    density = self.off_densities_[treat_val][t_bin_i]
                    # z_off → index dans [0, 2] par pas 0.01
                    z_idx = np.clip((z_off / 0.01).astype(int), 0, 200)
                    log_lik = np.where(
                        z_off > 2.0,
                        -30.0,
                        np.log(np.maximum(density[z_idx], 1e-30)) - np.log(scale * T_grid_clamped)
                    )
                    log_post += log_lik

                # Vraisemblance on_eq (seulement si off manquant)
                elif pd.notna(row.get("on_eq")):
                    time_known = pd.notna(row.get("time_since_intake_on"))
                    nboth_pos = int(row.get("nboth", 0)) > 0
                    key = (time_known, nboth_pos)
                    density = self.on_densities_.get(key, self.on_densities_[(False, False)])[t_bin_i]
                    z_on = float(row["on_eq"]) / (scale * T_grid_clamped)
                    z_idx = np.clip((z_on / 0.01).astype(int), 0, 200)
                    log_lik = np.where(
                        z_on > 2.0,
                        -30.0,
                        np.log(np.maximum(density[z_idx], 1e-30)) - np.log(scale * T_grid_clamped)
                    )
                    log_post += log_lik

                # Vraisemblance de début de traitement
                if treat_val == 1:
                    prev_treat = row.get("prev_treat_flag", None)
                    # prev_treat_flag doit être calculé avant appel
                    if prev_treat is not None and prev_treat == 0:
                        dur_v = float(row["dur"]) if pd.notna(row.get("dur")) else self.dur_median_
                        tb = _assign_bin_scalar(float(T_repr), _TREAT_T_EDGES)
                        db = _assign_bin_scalar(dur_v, _TREAT_DUR_EDGES)
                        if 0 <= tb < len(_TREAT_T_EDGES) - 1 and 0 <= db < len(_TREAT_DUR_EDGES) - 1:
                            p_treat = self.treat_start_prob_[tb, db]
                            log_post += np.log(max(p_treat, 1e-30))

            # Normaliser le log-posterior (stabilisation numérique)
            log_post -= log_post.max()
            posterior = np.exp(log_post)
            posterior /= posterior.sum()

            # Moments de T à chaque visite
            for visit_i, vi in enumerate(idx):
                x_v = float(x_pat[visit_i])
                x2_v = float(x_v ** 2 - x2_mean)

                # E[T] et Var[T]
                T_grid = (
                    _L_GRID[:, None, None]
                    + _S_GRID[None, :, None] * x_v
                    + _C_GRID[None, None, :] * x2_v
                )
                # E[T]
                et = float(np.sum(posterior * T_grid))
                et2 = float(np.sum(posterior * T_grid ** 2))
                vt = max(et2 - et ** 2, 0.0)
                post_mean[vi] = et
                post_sd[vi] = float(np.sqrt(vt))

        return np.stack([post_mean, post_sd], axis=1)

    def _add_prev_treat_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute prev_treat_flag (1 si visite précédente traitée, 0 sinon)."""
        out = df.copy()
        out["prev_treat_flag"] = (
            out.groupby("patient_id")["treat"].shift(1).fillna(0).astype(int)
        )
        return out


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
    sens_neighbors : bool
        Si True, la sensibilité au médicament utilise un a priori bayésien
        emprunté aux voisins en age_at_diagnosis dans le train.
    use_posterior : bool
        Si True, OneSidedPosterior est empilé hors pli (post_mean, post_sd).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 600,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        random_state: int = 0,
        sens_neighbors: bool = False,
        use_posterior: bool = False,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.sens_neighbors = sens_neighbors
        self.use_posterior = use_posterior

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
        df = _add_on_equivalent(
            df, self.drug_curve_,
            sens_neighbors=self.sens_neighbors,
            sens_ref=self.sens_ref_ if self.sens_neighbors else None,
        )
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
        extra_cols = []
        if self.sens_neighbors and "sens_prior" in df.columns:
            extra_cols.append("sens_prior")
        if self.use_posterior:
            if "post_mean" in df.columns:
                extra_cols.append("post_mean")
            if "post_sd" in df.columns:
                extra_cols.append("post_sd")

        feature_cols = BASE_COLS + POOL_COLS + TREAT_COLS + extra_cols + env_cols

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

        # Table de référence sens_neighbors
        if self.sens_neighbors:
            self.sens_ref_ = _fit_sens_reference(df, self.drug_curve_)
        else:
            self.sens_ref_ = None

        df = _add_on_equivalent(
            df, self.drug_curve_,
            sens_neighbors=self.sens_neighbors,
            sens_ref=self.sens_ref_,
        )

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

        # Bloc 7 : OneSidedPosterior empilé hors pli
        if self.use_posterior:
            post_mean_oof = np.zeros(len(df))
            post_sd_oof = np.full(len(df), 10.0)

            # GroupKFold interne à 3 plis sur patient_id
            gkf3 = GroupKFold(n_splits=3)
            groups = X_sorted["patient_id"].values
            df_with_target = df.copy()
            df_with_target["_target_"] = y_sorted.values

            for tr_idx, val_idx in gkf3.split(df, y_sorted, groups=groups):
                df_tr = df_with_target.iloc[tr_idx]
                df_val = df.iloc[val_idx]

                osp = OneSidedPosterior()
                osp.fit(df_tr, df_tr["_target_"])

                # Ajouter prev_treat_flag pour la vraisemblance de début de traitement
                df_val_pt = osp._add_prev_treat_flag(df_val)
                preds_val = osp.predict(df_val_pt)
                post_mean_oof[val_idx] = preds_val[:, 0]
                post_sd_oof[val_idx] = preds_val[:, 1]

            df["post_mean"] = post_mean_oof
            df["post_sd"] = post_sd_oof

            # Entraîner un OneSidedPosterior sur tout le train pour predict
            self.posterior_ = OneSidedPosterior()
            self.posterior_.fit(df_with_target, y_sorted)
        else:
            self.posterior_ = None

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

        # Bloc 7 : posterior depuis le modèle entraîné sur tout le train
        if self.use_posterior and self.posterior_ is not None:
            df_pt = self.posterior_._add_prev_treat_flag(df)
            post_preds = self.posterior_.predict(df_pt)
            df["post_mean"] = post_preds[:, 0]
            df["post_sd"] = post_preds[:, 1]

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


def build_learner(
    data_preview: pd.DataFrame | None = None,
    sens_neighbors: bool = False,
    use_posterior: bool = False,
) -> object:
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

    predictions = X.skb.apply(
        TrueOffEnvelopeRegressor(
            sens_neighbors=sens_neighbors,
            use_posterior=use_posterior,
        ),
        y=y,
    )
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
