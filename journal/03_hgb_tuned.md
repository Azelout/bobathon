# 03 — HGB accordé + ensemble multi-graines

**Statut :** en attente d'approbation

---

## Question / hypothèse

Le booster de 02 (`max_iter=600`, `max_leaf_nodes=31`) est peut-être sous-ajusté. Augmenter les itérations et la capacité des arbres, puis moyenner sur plusieurs graines, devrait faire passer la RMSE sous 2,9 sans changer l'architecture.

## Motivation

L'écart mesuré en validation croisée (RMSE 2,897) vs le score Kaggle (2,9) est petit mais réel. La feuille de route de `STRATEGIE.md` réserve `09_ensemble` (multi-graines + réglage léger) avant la finale. Ce palier est peu coûteux : pas de nouveau bloc, pas de nouvelle classe, juste des hyperparamètres différents et une moyenne de prédictions.

**Références :** `STRATEGIE.md` § 4.5 et § 5 (`09_ensemble`).

## Méthode

Changements par rapport à 02 :

| Hyperparamètre | 02 | 03 |
|---|---|---|
| `max_iter` | 600 | 1 000 |
| `max_leaf_nodes` | 31 | 63 |
| `min_samples_leaf` | 40 | 20 |
| `l2_regularization` | 1,0 | 0,5 |
| graines | 1 (0) | 3 (0, 1, 2) — moyenne des prédictions |

**Stratégie d'ensemble :** entraîner 3 instances de `TrueOffEnvelopeRegressor` (graines 0, 1, 2) sur les mêmes données ; la prédiction finale est la moyenne. Le post-traitement par patient (bloc 6) est appliqué sur cette moyenne.

**Architecture inchangée :** mêmes blocs 1 à 6, même estimateur, même graphe skrub.

## Risques

- Si le booster était déjà bien calibré à 600 itérations, augmenter à 1 000 pourrait entraîner un léger surapprentissage. Contrôle : comparer RMSE sans enveloppe (doit rester proche de 3,54 ± 0,05).
- La moyenne de 3 graines triple le temps de calcul (fit), mais garde predict identique.
- `min_samples_leaf=20` avec `max_leaf_nodes=63` peut surappendre sur les patients rares. Si la RMSE se dégrade, revenir à `min_samples_leaf=40`.

## Validation

- `skore.evaluate` sur le graphe, 5 plis `GroupKFold` sur `patient_id`.
- Critère d'acceptation : RMSE < 2,9 (tous les plis inférieurs à 3,0).

## Soumission

1. `EstimatorReport` sur le premier pli → poussé sur Hub, clé `03_hgb_tuned`.
2. Entraîner sur tout le train, prédire `X_test` → `submissions/03_hgb_tuned.csv`.

---

## Bloc de statut

- **State :** approved
- **Approved by user on :** 2026-10-14
- **State :** done
- **Headline result :** RMSE = 2,827 ± 0,050 (GroupKFold 5 plis) — https://skore.probabl.ai/thr_bob/bobathon-esilv/cross-validations/80138
- **Implication for next iteration :** Passe clairement sous 2,9. Soumettre sur Kaggle pour confirmer. Si score public < 2,9, B7 (modèle bayésien empilé) peut descendre jusqu'à ≈ 2,84.

## Stratégie de sourcing

`my-pick` (synthèse de la feuille de route § 5 de `STRATEGIE.md`, choix utilisateur).
