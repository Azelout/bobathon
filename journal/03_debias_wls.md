# 03 — Débiaisage mesure + WLS parabolique par patient

**Statut :** done
**Approved by user on :** 2026-10-14

---

## Résultat

**Architecture finale retenue :** `TrueOffEnvelopeRegressor` avec HGB accordé (`max_iter=1000`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=0.5`) + ensemble 3 graines.

Ceci est l'implémentation de 03_hgb_tuned, qui a déjà été exécutée et soumise (RMSE CV = 2,827 ± 0,050).

## Pistes testées et rejetées

| Piste | RMSE fold 0 | Verdict |
|---|---|---|
| WLS parabole quantile sur estimations débiaisées | 11,98 | Bruit individuel std=25 pts — trop élevé |
| Enveloppe HGB + débiaisage (tbin, rater) en remplacement des blocs 2-3 | 3,29 | Dégrade vs 02 |
| Stacking HGB + WLS (OOF GroupKFold 3) | 2,862 | WLS contribue ~0 (alpha≈0) |

## Conclusion

Les estimations individuelles débiaisées (std=25 pts, bruit relatif 68 %) sont trop bruitées pour servir de base à une parabole par patient avec 4–8 visites. Le HGB avec enveloppe est déjà la meilleure agrégation possible de ce signal bruité. La voie vers 2,3 nécessite soit un modèle de vraisemblance bayésien complet (Bloc 7), soit des features supplémentaires non encore exploitées.

## Prochaine piste recommandée

Bloc 7 (STRATEGIE.md § 4.7) : modèle de vraisemblance par patient empilé hors pli. Gain attendu : 2,827 → ~2,78 (≈ 0,05 pt). Coût : ~25 min par validation.
