# 02 — True OFF à enveloppe

**Statut :** en attente d'approbation  
**Clé skore :** `02_true_off_envelope`  
**RMSE attendue :** ≈ 2,90 (fourchette 2,82 – 2,95 par pli)

---

## Contexte

`STRATEGIE.md` sections 1.2, 4 et 11 décrivent un estimateur unique qui implémente tous les blocs de la recette recommandée :

- Bloc 1 : variables de visite et de traitement, sans cible.
- Bloc 2 : conversion de `on` en `on_eq` via une courbe d'effet du médicament et une sensibilité par patient.
- Bloc 3 : mise à l'échelle par évaluateur (`off_c`, `on_c`), corrections apprises dans `fit`.
- Bloc 4 : droite pondérée par patient + résumés + variables d'enveloppe (90 colonnes).
- Bloc 5 : `HistGradientBoostingRegressor` sur toutes les variables.
- Bloc 6 : post-traitement par patient (parabole → isotonique → bornes 0-132).

La section `Corrections after the independent review (v4)` de `data/eda.md` confirme le modèle de bruit à sens unique et la quasi-déterminabilité de `on`/`off`.

---

## Décisions de conception

### Estimateur scikit-learn personnalisé

- Classe `TrueOffEnvelopeRegressor(BaseEstimator, RegressorMixin)`.
- `fit` et `predict` trient en interne par `patient_id` puis `age`, et restituent l'ordre d'origine.
- Tout ce qui utilise la cible est calculé dans `fit` uniquement.

### Graphe skrub DataOps

- Pas de `TableVectorizer` ni de `tabular_pipeline` : le modèle a besoin des colonnes brutes, dont `patient_id`.
- L'estimateur est connecté via `.skb.apply(TrueOffEnvelopeRegressor(), ...)`.
- X est marqué avec `cv=GroupKFold(n_splits=5)` et `split_kwargs={"groups": patient_id}`.

---

## Bloc 1 — Variables de visite et de traitement (sans cible)

| Variable | Calcul |
|---|---|
| `dur` | `age − age_at_diagnosis` |
| `t_rel` | `age` − âge à la première visite du patient |
| `vis` | rang de visite (0-based) par patient |
| `nv` | nombre de visites du patient |
| `span` | âge à la dernière visite − âge à la première visite |
| `rater` | entier de `rater_id` (catégorie pour HGB) |
| `coh` | 1 si cohorte B |
| `gene_c` | GBA+ → 0, LRRK2+ → 1, No Mutation → 2, OTHER+ → 3, manquant → −1 |
| `hbin` | tranche de `time_since_intake_on` (rempli à −1), bornes [−2, −0,5, 0,25, 0,5, 0,75, 1, 1,5, 2, 3, 4, 7] |
| `hb` | tranche plus fine, bornes [−2, −0,5, 0,1, 0,2, 0,3, 0,5, 0,75, 1, 1,25, 1,5, 2, 2,5, 3, 4, 7] |
| `treat` | 1 si `on`, `ledd` ou `time_since_intake_on` présent |
| `age_first_treat` | âge à la première visite traitée du patient |
| `t_since_treat` | `age − age_first_treat` |
| `t_to_lastuntr` | `age` − âge à la dernière visite non traitée |
| `dur_treat` | âge au premier traitement − `age_at_diagnosis` |
| `n_treat` | nombre de visites traitées du patient |

---

## Bloc 2 — Conversion de `on` (sans cible)

Appris dans `fit` :

1. **Courbe d'effet** : médiane de log(`on`/`off`) par tranche `hb`, sur les visites doubles (`off` ≥ 8 et `on` ≥ 1).
2. **Sensibilité par patient** : somme des écarts à la courbe divisée par (nombre d'écarts + 1). Vaut 0 sans visite double. `nboth` = nombre de visites doubles.
3. `on_eq` = `on` × exp(−(courbe[`hb`] + sensibilité)).

---

## Bloc 3 — Corrections avec la cible (apprises dans `fit`)

1. **Droite par évaluateur** : `off` = a × cible + b → `off_c` = (`off` − b) / a.
2. **Effet ON** : médiane de log(max(`on`, 0,5) / cible) par `hbin`, sur les visites avec `on` et cible > 3. Plus un décalage par évaluateur (médiane des écarts). → `on_c` = `on` / exp(effet + décalage, ou 0 si inconnu).
3. **Bord par évaluateur** : `q95` = 95e centile de `off`/cible, sur les visites avec `off` et cible > 15.

---

## Bloc 4 — Variables par patient

### Droite pondérée

- Sur toutes les valeurs `off_c` et `on_c` du patient, en fonction de `t_rel` centré.
- Niveau de référence m = médiane de ces valeurs, ou 30 si aucune.
- Poids : 1/(2 + 0,17 m)² pour `off_c`, 1/(0,45 m)² pour `on_c`.
- Pente rapprochée de 2,5 avec pénalité (1/1,35²) × poids moyen. Pente bornée à 0.
- Sorties : `pool`, `pslope`, `pres`.

### Résumés

- `n_off`, `n_on`, médiane de `off_c` et `on_c` par patient.
- `off_c` de la visite précédente et suivante ayant une valeur.

### Enveloppe (90 colonnes)

- `u_off` = `off`/`q95`, `u_on` = `on_eq`/`q95`, `u_any` = `u_off` si présent, sinon `u_on`.
- Pour chaque visite v, chaque série (3) et chaque pente s ∈ {0, 1,2, 2,4, 4, 6} : pour chaque mesure j du patient, calculer `u_j + s × (âge_v − âge_j)`, trier décroissant, garder les rangs 1 à 6.
- 3 × 5 × 6 = 90 colonnes, NaN si moins de mesures disponibles.

---

## Bloc 5 — Gradient boosting

`HistGradientBoostingRegressor` :

| Hyperparamètre | Valeur |
|---|---|
| `learning_rate` | 0,05 |
| `max_iter` | 600 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 40 |
| `l2_regularization` | 1 |
| `early_stopping` | False |
| `categorical_features` | `[rater]` (par liste d'indices ou masque) |
| `random_state` | 0 |

Variables d'entrée : colonnes brutes numériques (`sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_off`, `time_since_intake_on`, `on`, `off`) + toutes les variables des blocs 1 à 4. Exclure `patient_id` et `Index`.

---

## Bloc 6 — Post-traitement par patient

1. Ajuster une parabole (degré min(2, nv − 1)) en `age` centré sur les prédictions du patient, la substituer.
2. `IsotonicRegression` par patient sur les prédictions triées par âge.
3. Borner entre 0 et 132.

---

## Validation

- `skore.evaluate` sur le graphe, 5 plis `GroupKFold` sur `patient_id`.
- RMSE attendue ≈ 2,90 (fourchette 2,82 – 2,95 par pli).
- Contrôle : sans colonnes d'enveloppe → RMSE ≈ 3,54. Si écart > 0,05, vérifier l'apprentissage dans `fit`, l'ordre des lignes et le calcul de l'enveloppe.

---

## Tests (dans `tests/smoke/test_02_true_off_envelope.py`)

1. La prédiction d'un patient est identique seul ou avec d'autres patients.
2. Mélanger les lignes ne change pas les prédictions.
3. Les prédictions sont croissantes par patient et comprises entre 0 et 132.
4. La prédiction sur `X_test` a 11 013 lignes finies, avec le même index que `sample_submission.csv`.

---

## Soumission (après validation)

1. Construire un `EstimatorReport` sur le premier pli `GroupKFold` (séparation groupée).
2. Pousser sur le Hub, projet `bobathon-esilv`, clé `02_true_off_envelope`.
3. Entraîner sur tout le train, prédire `X_test`, écrire `submissions/02_true_off_envelope.csv` (`Index,target`).

---

## Critères d'acceptation

- [ ] RMSE de validation ≈ 2,90 (tous les plis entre 2,82 et 2,95).
- [ ] RMSE sans enveloppe ≈ 3,54 (±0,05).
- [ ] Les 4 tests passent.
- [ ] `submissions/02_true_off_envelope.csv` a 11 013 lignes, index correct.
- [ ] URL du rapport Hub disponible.
