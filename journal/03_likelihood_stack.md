# 03 — Likelihood stack

**Date :** 2026-10-14  
**Base :** `02_true_off_envelope` (RMSE = 2,897)  
**Objectif :** ≈ 2,82 en validation GroupKFold 5 plis

---

## Deux ajouts à `TrueOffEnvelopeRegressor`

### 1. `sens_neighbors=True` — sensibilité empruntée aux voisins en âge au diagnostic

**Observation :** des patients avec le même `age_at_diagnosis` ont une corrélation de sensibilité de 0,65 ; au-delà de 6 mois d'écart, elle tombe à 0,08. Les courbes de cible, elles, ne se ressemblent pas.

**Mécanisme :**
- Dans `fit`, on garde une table des patients d'entraînement avec au moins une visite double : résidu moyen `r_j` (somme / count) et poids `w_j = count / (count + 1)`.
- Pour un patient cible d'âge au diagnostic `a`, a priori bayésien : `Σ w_j·k_j·r_j / (Σ w_j·k_j + 0.5)` avec `k_j = exp(−½·((a_j − a)/0.05)²)` (noyau gaussien, σ = 0,05 an = 18 jours), patient lui-même exclu.
- Sensibilité finale : `(somme_résidus + 2·a_priori) / (count + 2)` au lieu de `somme / (count + 1)`.
- Variable supplémentaire `sens_prior` ajoutée au boosting.
- En `predict` : la référence reste la table d'entraînement seule → isolation par patient préservée.

**Gain attendu :** faible mais gratuit (pas de cible utilisée).

### 2. `use_posterior=True` — `OneSidedPosterior` empilé hors pli

**Classe `OneSidedPosterior`** (appris dans `fit`, `predict` hors pli) :

| Composant | Détail |
|---|---|
| Grille courbe | L ∈ [0, 130] × S ∈ [0, 12] × C ∈ [−0.3, 0.3] (pas 0.5/0.25/0.1) |
| Vraisemblance `off` | Densité empirique de `z = off/(échelle × T)` par tranche de T et par `treat`, lissée |
| Vraisemblance `on_eq` | Même principe, conditionné sur `time_since_intake_on` connu × `nboth > 0` |
| Vraisemblance traitement | P(`treat`=1) par tranche de T et de `dur`, table de Laplace |
| A priori L | Régression linéaire sur [1, dur, dur², part traitée] |
| A priori S | Histogramme des pentes linéaires des patients |
| A priori C | Histogramme des courbures quadratiques (≥ 5 visites) |
| Sorties | `post_mean`, `post_sd` de T à chaque visite |

**Empilement :** `GroupKFold(3)` interne sur `patient_id` ; un `OneSidedPosterior` par pli d'entraînement → `post_mean`, `post_sd` hors pli → passés au boosting.

---

## Protocole

1. Essai bug-check sur 300 patients (~2 min).
2. Validation complète `skore.evaluate` (5 plis GroupKFold, ~15 min).
3. **Seuil d'arrêt :** RMSE > 2,88 → montrer les scores par pli avant toute soumission.
4. Tests smoke `test_02_true_off_envelope.py` avec les deux options à `True`.
5. `EstimatorReport` pli 1 → Hub (clé `03_likelihood_stack`).
6. Fit complet + `submissions/03_likelihood_stack.csv`.
