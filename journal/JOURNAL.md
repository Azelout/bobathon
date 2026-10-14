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
- **Last experiment:** n/a - not yet run
- **Last result:** n/a

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-11
  - env manager: pip+venv - recorded: 2025-07-11
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: <none> - recorded: <YYYY-MM-DD>
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: <local | hub | mlflow> - recorded: <YYYY-MM-DD>
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: <KFold | StratifiedKFold | GroupKFold | TimeSeriesSplit | other> - recorded: <YYYY-MM-DD>

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
| `02_true_off_envelope` | TrueOffEnvelopeRegressor — blocs 1-6, enveloppe 90 cols | approved | RMSE attendue ≈ 2,90 | [design note](02_true_off_envelope.md) |

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
