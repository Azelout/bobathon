# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's Disease UPDRS score prediction (hackathon clinical dataset)
- **Goal:** Predict the debiased true OFF MDS-UPDRS motor score for every visit (regression)
- **Last experiment:** `03_hgb_tuned` — done
- **Last result:** RMSE = 2,827 ± 0,050 (GroupKFold 5 plis) — https://skore.probabl.ai/thr_bob/bobathon-esilv/cross-validations/80138

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-11
  - env manager: pip+venv - recorded: 2025-07-11
  - agent feature: installed - recorded: 2025-07-11
  - optional features: none - recorded: 2025-07-11
  - package name (`src/<pkg>/`): parkinson - recorded: 2025-07-11
  - skore mode: hub - recorded: 2025-07-11
  - skore hub workspace: bobathon-esilv - recorded: 2025-07-11
  - skore mlflow tracking uri: n/a - recorded: 2025-07-11
  - student prior: comfortable - recorded: 2025-07-11
  - CV splitter family: GroupKFold - recorded: 2025-07-11

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done v2 - 2026-10-11 (v1 from 2025-07-11 superseded)
- **Summary:** 44,590 train visits from 5,576 patients, 11,013 test visits from 1,395 patients, no patient overlap → `GroupKFold(groups=patient_id)`. Target is the true OFF score (0–109.5, mean 37.5, std 16.5, symmetric). Within each patient the target never decreases and is a near-perfect quadratic in age (residual RMSE 0.21). `off` is a biased noisy reading (−5.4 mean bias, RMSE 10.2), `on` reads ~22 lower; both are inputs, not leakage. Raters carry stable offsets (−10.5 to +2.2). Missingness follows disease stage: early visits lack `on`/`ledd`, later visits lack `off`. Dummy RMSE 16.5.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `02_true_off_envelope` | TrueOffEnvelopeRegressor — blocs 1-6, enveloppe 90 cols | done | RMSE = 2,897 ± 0,054 — submission `submissions/02_true_off_envelope.csv` | [design note](02_true_off_envelope.md) |
| `03_hgb_tuned` | HGB accordé (iter=1000, leaves=63) + ensemble 3 graines | done — meilleur résultat courant, submission soumise | RMSE = 2,827 ± 0,050 — submission `submissions/03_hgb_tuned.csv` | [design note](03_hgb_tuned.md) |
| `03_debias_wls` | Pivot vers nouvelle modélisation : WLS/stacking testés, non concluants — meilleur = 03_hgb_tuned | done | RMSE 2,827 ± 0,050 (même que 03_hgb_tuned) | [design note](03_debias_wls.md) |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
| <!-- B1 --> | <!-- "investigate target-bin>0.95 residual bias via target transform" --> | <!-- `skore:01_baseline` --> |
| B1 | Plis `GroupKFold` à 5 plis sauvegardés et réutilisés | [STRATEGIE.md](STRATEGIE.md) section 3 |
| B2 | Variables de visite et de date de traitement | [STRATEGIE.md](STRATEGIE.md) section 4.1 |
| B3 | Conversion de `on` en équivalent `off` : courbe d'effet et sensibilité par patient, sans cible | [STRATEGIE.md](STRATEGIE.md) section 4.2 |
| B4 | Échelle par évaluateur au 95e centile, apprise dans le pli | [STRATEGIE.md](STRATEGIE.md) section 4.3 |
| B5 | Variables d'enveloppe : rangs 1 à 6 des mesures ramenées selon 5 pentes, gain attendu d'environ 0,7 | [STRATEGIE.md](STRATEGIE.md) section 4.4 |
| B6 | Projection parabolique, puis isotonique, puis bornes, par patient | [STRATEGIE.md](STRATEGIE.md) section 4.6 |
| B7 | Modèle de vraisemblance à bruit à sens unique, empilé hors pli | [STRATEGIE.md](STRATEGIE.md) section 4.7 |
