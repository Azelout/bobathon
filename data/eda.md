<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Every number below is reproduced by that script.
"Modelling implications" are candidate suggestions, not final decisions.
-->

# EDA v2: predicting the true OFF MDS-UPDRS motor score

_Generated from `data/eda.py` on 2026-10-11, with a v3 section on the measurement mechanism and v4 corrections after an independent review. Replaces v1, which misread `on`/`off` as leakage and the target as a composite score._

## Dataset at a glance

- **Tables:** `X_train.csv` (12 features), `y_train.csv` (`target`), `X_test.csv` (same 12 features), `sample_submission.csv` (`Index,target`, same index as `X_test`).
- **Rows:** 44,590 train visits from 5,576 patients; 11,013 test visits from 1,395 patients.
- **Target:** `target`, the debiased "true OFF" motor score. Regression.
- **Overlap:** 0 patients shared between train and test. All 50 test raters appear in train.
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)
- **Shift:** train and test feature means match closely. No covariate shift visible.

## Patients and visits (workbook Q1)

- Every patient has between 4 and 12 visits. Median is 7, mean is 8.
- The distribution is bimodal: 1,601 patients have exactly 4 visits and 2,038 have exactly 12.
- Visits are about one year apart (median 0.9 years, range 0.2 to 5.7).
- Rows are already sorted by `age` within each patient, with no duplicate ages.
- `cohort`, `sexM`, `gene` and `age_at_diagnosis` are constant within a patient. They are patient-level attributes.
- Each patient is seen by 1 to 3 raters, usually 3. Every rater works in both cohorts.

## Missingness (workbook Q2, Q5)

| Column | Missing train | Missing test |
|---|---|---|
| `time_since_intake_off` | 78.8 % | 79.1 % |
| `time_since_intake_on` | 46.4 % | 47.8 % |
| `off` | 42.4 % | 40.8 % |
| `ledd` | 36.6 % | 38.6 % |
| `gene` | 32.4 % | 32.0 % |
| `on` | 29.6 % | 31.2 % |
| `age_at_diagnosis` | 5.2 % | 4.6 % |

- **No visit lacks both scores.** 28 % of visits have both `on` and `off`.
- **Missingness is structured, not random.** It follows disease stage:
  - `on` is missing on 78 % of first visits and 3 % of twelfth visits. Untreated patients have no ON state.
  - `ledd` follows the same pattern, from 80 % to 12 % missing. A missing `ledd` almost always means a missing `on`.
  - `off` goes the other way, from 13 % to 58 % missing. OFF exams get skipped as patients are treated.
- **Missing `ledd` marks earlier disease.** Those visits average 3.4 years since diagnosis and a target of 25.6. Visits with `ledd` average 7.3 years and 44.3.
- **Cohort B has more gaps.** `ledd` is missing on 42.6 % of B visits against 35.9 % of A. `gene` is missing on 44.4 % against 30.9 %. Cohort B patients are also earlier in disease, by about two years.
- **Intake delays only exist alongside their score.** `time_since_intake_on` is never present without `on`, and the same holds for `off`.
- `age_at_diagnosis` missing makes years-since-diagnosis unavailable for 5 % of visits.

## Target and its trajectory (workbook Q3)

- **Range** 0 to 109.5 out of a theoretical 0 to 132. Mean 37.5, median 37.3, standard deviation 16.5.
- **Shape** is nearly symmetric, with skew 0.05. 173 visits sit exactly at 0. No transform is needed.
- **Progression drives it.** Mean target rises from 24 in the first year after diagnosis to 56 after 15 years.
- **Within a patient the target never decreases.** This holds for 100 % of patients.
- **Each patient's target is a smooth curve of age.** A per-patient quadratic leaves an in-sample residual RMSE of 0.21 points, and a straight line 1.19. Corrected for degrees of freedom, the quadratic residual is about 0.24 to 0.34.
- Typical progression is 2.4 points per year, ranging from 0 to 15.
- Starting level varies a lot between patients, with a standard deviation of 14.6 at the first visit.

The prediction problem is therefore about estimating each patient's smooth curve. Visit-to-visit noise in the target is almost nil.

## Measured OFF and ON versus the true OFF (workbook Q4)

| Feature | Pearson r with target |
|---|---|
| `off` | 0.871 |
| `on` | 0.669 |
| years since diagnosis | 0.542 |
| `age` | 0.310 |
| `ledd` | 0.298 |
| `age_at_diagnosis` | 0.133 |
| `sexM`, intake delays | about 0 |

- **`off` is a biased, noisy reading of the target, not a copy.** It reads 5.4 points low on average, with a standard deviation of 8.7. Its RMSE against the target is 10.2.
- **The OFF error grows with severity.** Bias goes from about −2 for targets under 15 to about −8 above 60. The spread grows from 3.5 to 11.6. The error is roughly multiplicative, with `off` around 0.88 times the target.
- **`on` reads 21.8 points below the target.** ON is about 60 % of OFF when both are measured, consistent with the 50 to 100 % improvement in the clinical context.
- `on` and `off` are inputs, not leakage. They are the noisy measurements the task asks to debias.

## Drug timing

- `time_since_intake_on` is a short delay of 0 to 6.3 hours, median 1.6.
- `time_since_intake_off` is a long delay of 6.9 to 24.7 hours, median 13.8.
- **OFF reads lower on treated visits, but the drug is not the main cause.** Treated visits read about 7.4 points low and untreated ones about 3.6. The v4 section shows the same downward skew in patients who are never treated, so the gap mostly reflects higher severity on treated visits, not residual levodopa.
- **ON relief builds for about 2 hours, then plateaus.** The mean log ratio of `on` to target goes from −0.22 in the first 15 minutes to about −0.87 after 2 hours, and stays there up to 7 hours. It does not depend on `ledd` or disease duration. Measured against `off` at the same visit, the drug effect is much cleaner: see v4.
- Raw intake delays have near-zero linear correlation with the target. Their signal is in how they bend the ON and OFF readings, so they need interactions or a tree model.

## Rater bias

- **Raters have stable personal offsets.** Mean OFF error per rater ranges from −10.5 for R03 to +2.2 for R50.
- **The bias is real, not noise.** Rater offsets estimated on two disjoint patient halves correlate at 0.96.
- **It is the same bias on both scales.** A rater's ON offset and OFF offset correlate at 0.975.
- Removing the rater mean shrinks the OFF error spread only a little, from 8.7 to 8.3. Rater bias is one of several error sources.

## Measurement mechanism (v3)

These numbers come from corrections fitted in-sample with the target. They show what is achievable, not a model score.

- **Scores are integers.** `on` and `off` are whole numbers, and 618 `off` values are exactly 0. The measurement is rounded and clipped at 0.
- **Raters scale, not just shift.** Per rater, `off` is roughly a fixed fraction of the target. That fraction ranges from 0.69 to 1.12, with a median of 0.86.
- **OFF noise is independent across visits.** After removing the scale and rater effects, consecutive errors of a patient correlate at −0.002. The share of error variance between patients equals what pure chance predicts, 0.20.
- **OFF noise grows with severity.** Its standard deviation goes from 3.5 for targets under 15 to 10.3 above 60.
- **ON looks noisy against the target, but not against OFF.** Its log ratio to the target has a standard deviation of about 0.44. Most of that is the visit-level error it shares with `off`. Read relative to `off`, ON is nearly deterministic: see v4.
- **Patient covariates barely predict the curve.** Age at diagnosis, sex, cohort and gene explain 3.4 % of the level at diagnosis and 0.6 % of the slope.
- **Cohort B is mostly short follow-ups.** 67 % of cohort B patients have exactly 4 visits, against 19 % in cohort A.

**Pooling diagnostic.** Each measurement is first corrected for rater and drug effects. Then a curve in age is fitted per patient through all corrected measurements.

| Proxy | RMSE |
|---|---|
| Corrected `off`, visit level | 8.90 |
| Corrected `on`, visit level | 13.88 |
| Pooled per patient, constant | 9.14 |
| Pooled per patient, linear | 6.58 |
| Pooled per patient, linear, visits without `off` | 7.67 |
| Pooled per patient, quadratic | 7.26 |

- Linear pooling beats every visit-level proxy, because the noise is independent across visits.
- A quadratic fits worse than a line, even though the truth is quadratic. With 4 to 12 noisy points, the extra parameter adds more noise than it removes bias.
- 14.3 % of pooled lines slope downward, while the true curve never does.

## Corrections after the independent review (v4)

An independent review found that the v3 noise model was wrong. These numbers are reproduced by the v4 cell of `data/eda.py`.

**The OFF error is one-sided.** Each reading is divided by its rater's 95th percentile of `off` over target, on visits with a target above 15.

| Quantile of the scaled ratio | Value |
|---|---|
| 5 % | 0.31 |
| 25 % | 0.58 |
| 50 % | 0.76 |
| 75 % | 0.89 |
| 95 % | 1.00 |
| 99 % | 1.09 |

- Readings pile up just under a sharp upper edge and trail far downward. The skew within raters is −0.73.
- For this kind of noise, a patient's highest readings locate the curve better than the mean. Averaging, as in the v3 pooling diagnostic, is inefficient.
- **The skew is not a drug effect.** Patients never treated have a median scaled ratio of 0.73, against 0.76 for the others.

**ON is OFF times a nearly fixed drug factor.** This uses visits with both readings, with `off` of at least 8 and `on` of at least 1, and no target at all.

- The median of log(`on`/`off`) goes from 0 in the first 6 minutes after intake to −0.58 from 1.5 hours on, and stays flat up to 7 hours. Visits without intake timing sit at −0.51.
- Around that curve, the residual has a standard deviation of only 0.19, and 0.095 within a patient.
- **The drug sensitivity is patient-specific.** 72 % of the residual variance lies between patients, against 24 % expected by chance. A patient's odd and even visits agree at 0.82.
- 42 % of visits have only `on`. Converted through this curve and the patient's sensitivity, they become almost as informative as `off`.

**What this changes in v3.** The claims that residual levodopa lowers OFF, and that ON is too noisy to weigh much, are withdrawn. The pooling diagnostic still shows that pooling helps, but its method is not the right one.

## Patient-level attributes

- **`cohort`:** A has 39,636 visits, B has 4,954. B targets are lower, mostly because B patients are earlier in disease.
- **`gene`:** No Mutation and missing are the largest groups. OTHER+ patients are diagnosed younger, at 51 on average. LRRK2+ patients are diagnosed later, at 59. Differences in mean target are small, at most about 4 points.
- **`sexM`:** no association with the target.

## Reference errors

These are not models. They show how far simple proxies are from the target.

| Proxy | RMSE |
|---|---|
| Dummy mean | 16.50 |
| Raw `off`, where present | 10.20 |
| Patient mean of `off`, all visits | 12.03 |

Naively averaging a patient's `off` readings is worse than the raw reading, because it ignores progression. Pooling has to follow the curve over time.

## What not to feed the model (workbook Q6)

- **`patient_id`:** an identifier with no overlap with test. Use it only for grouping and for building per-patient features.
- **`Index`:** row id only.
- **`rater_id`:** keep it. Its bias is stable and every test rater is seen in train. Target-encode it or let a tree split on it.
- **`on`, `off` and intake delays:** keep them. They are the signal to debias, not leakage.

## Modelling implications

- **Validate with `GroupKFold(groups=patient_id)`.** The Kaggle holdout is by patient, so random row splits will overstate scores.
- **Add years since diagnosis**, computed as `age - age_at_diagnosis`. It correlates at 0.54 with the target, far more than `age`.
- **Build per-patient features from all visits.** The target is a smooth monotone curve per patient. Each test patient's visits are all in `X_test`, so features like the patient's fitted OFF level and slope over age are legitimate.
- **Keep missingness as signal.** Missing `ledd` and `on` mark early, untreated disease. `HistGradientBoosting` handles NaN natively, and missing-indicator columns help linear models.
- **Correct for rater bias** with target encoding of `rater_id` or a per-rater offset on `off` and `on`.
- **Convert ON through OFF, not through the target.** Use an intake-time curve of log(`on`/`off`) and a per-patient drug sensitivity, both computed without the target.
- **Pool with the upper envelope, not the mean.** The OFF error is one-sided, so a patient's highest rater-scaled readings locate the curve best.
- **Expect heteroscedastic errors.** OFF noise grows with severity. Check residuals by target level.
- **Metric:** RMSE. The dummy floor is 16.5. Raw `off` is 10.2 where it exists.

## Open questions

1. Could a monotone constraint or post-hoc isotonic smoothing per patient enforce the never-decreasing trajectory?
2. Is the benchmark heuristic in `docs/CONTEXT.md` reproducible from these features, and what is its RMSE?
3. Does the bimodal visit count of 4 or 12 visits reflect two study designs, and does it interact with `cohort`?
