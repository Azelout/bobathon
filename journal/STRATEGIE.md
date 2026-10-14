# Stratégie pour le meilleur score : prédire le vrai score OFF

_Version 2, rédigée le 2026-10-11. Elle remplace la version 1 après une relecture indépendante. Les chiffres de RMSE viennent d'expériences en validation croisée à 5 plis groupés par patient, où tout ce qui utilise la cible est appris sur le pli d'entraînement. Ces expériences sont des essais exploratoires : l'équipe doit les réimplémenter dans Bob avant toute soumission. Leurs scripts sont dans `scratch/review/`, avec le rapport de relecture `REVIEW.md`, et dans `scratch/analysis/`. Une implémentation de référence propre de toute la stratégie se trouve dans `scratch/pipeline/`, voir la section 11. Ce dossier est exclu de git, il n'existe donc que sur cette machine._

## 0. Synthèse

Le défi consiste à **retrouver une courbe cachée par patient** à partir de mesures `off` et `on` bruitées. La version 1 avait bien identifié ce cadre. Elle s'était trompée sur un point décisif : la forme du bruit.

**Le bruit est à sens unique.** Une mesure est soit proche de la vraie valeur, soit trop basse, et presque jamais trop haute. Pour un tel bruit, les mesures les plus hautes d'un patient localisent sa courbe bien mieux que la moyenne. Toute la stratégie découle de ce fait.

La recette recommandée :

1. **Variables de visite et de traitement**, sans cible.
2. **Conversion de `on` en équivalent `off`**, par une courbe d'effet du médicament et une sensibilité propre à chaque patient, sans cible.
3. **Mise à l'échelle par évaluateur**, apprise sur le pli d'entraînement.
4. **Variables d'enveloppe** : pour chaque visite, les plus hautes mesures du patient ramenées à l'âge de cette visite.
5. **Gradient boosting** sur l'ensemble.
6. **Post-traitement par patient** : projection sur une parabole, puis projection isotonique, puis bornes.
7. **En option**, un modèle de vraisemblance par patient, empilé comme variable supplémentaire.

| Étape cumulée | RMSE en validation groupée |
|---|---|
| Moyenne constante | 16,50 |
| Boosting sur colonnes brutes | 7,77 |
| Plus résumés par patient | 4,39 |
| Plus corrections apprises et droite pondérée par patient | 3,81 |
| Plus `on` converti, sensibilité par patient, dates de traitement | 3,63 |
| Plus variables d'enveloppe | 2,95 |
| Plus projection parabolique puis isotonique | 2,89 |
| Plus modèle de vraisemblance empilé | 2,85 |
| Meilleur score public Kaggle au 2026-10-11 | 2,196 |

Il reste environ 0,65 point d'écart avec le meilleur score public. La section 9 liste les pistes pour le réduire.

---

## 1. Ce que le problème est vraiment

### 1.1 Les faits établis

| Fait | Preuve | Conséquence |
|---|---|---|
| Les patients du test sont absents du train | 0 patient commun, aucun doublon de trajectoire | Valider par patient |
| La cible ne baisse jamais chez un patient | 100 % des patients, 93,6 % strictement croissants | Contrainte monotone |
| La cible est une courbe lisse de l'âge | Résidu d'une parabole par patient de 0,24 à 0,34 point, corrigé des degrés de liberté | Peu de paramètres par patient |
| **Le bruit de `off` est à sens unique** | Rapport `off` sur cible : bord supérieur net par évaluateur, longue traîne vers le bas, asymétrie de −0,67 | Exploiter les mesures hautes, pas la moyenne |
| Chaque évaluateur a son échelle | Le bord supérieur dépend de l'évaluateur. Le reste de la forme du bruit est commun | Mettre à l'échelle par évaluateur |
| Le bruit est indépendant d'une visite à l'autre | Corrélation entre erreurs successives de −0,04 | Plusieurs visites apportent chacune de l'information |
| Le bruit est commun à `on` et `off` dans une même visite | Corrélation de 0,76, ou 0,82 en logarithme | Une visite à deux mesures ne vaut pas deux visites |
| **`on` est `off` multiplié par un effet du médicament presque déterministe** | Selon le délai de prise, le log de `on` sur `off` ne varie que de 0,11 chez un même patient | Convertir `on` par rapport à `off`, pas par rapport à la cible |
| L'effet du médicament monte pendant 2 heures puis plafonne | Log de `on` sur `off` : 0 avant 6 minutes, −0,58 de 1,5 à 7 heures | Courbe d'effet par tranches de délai |
| La sensibilité au médicament est propre au patient | 72 % de la variance résiduelle est entre patients, contre 23 % attendus par hasard. Visites paires et impaires concordent à 0,71 | Estimer une sensibilité par patient |
| **Le début du traitement renseigne sur la sévérité** | Aucun traitement ne commence sous une vraie valeur de 18, ni dans les 6 mois après le diagnostic. Un traitement commencé ne s'arrête jamais | Variables de date de traitement |
| `off` est toujours mesuré avant traitement | 0 % de manque avant, environ 60 % après | Le manque de `off` indique un patient traité |
| Les caractéristiques fixes prédisent peu la courbe | 3,4 % du niveau, 0,6 % de la pente | Ne pas investir dans `gene`, `sexM`, `cohort` |
| La dose suit le temps, pas la sévérité | Corrélation entre pente de dose et pente de cible : 0,003 | Pas de variable de sévérité tirée de `ledd` |

### 1.2 Le modèle de mesure

Pour chaque patient *i*, il existe une vraie courbe croissante T_i(âge). On l'observe à travers :

- **`off`** ≈ échelle de l'évaluateur × T × *u*, où *u* vaut au plus environ 1 et descend souvent bien plus bas. La valeur est arrondie à l'entier.
- **`on`** ≈ `off` de la même visite × effet du médicament selon le délai × sensibilité propre au patient. `on` hérite donc du même *u* que `off` à cette visite.

La sous-estimation existe aussi chez les patients jamais traités, où le rapport médian vaut 0,64. Elle ne vient donc pas du médicament résiduel, contrairement à ce qu'affirmait la version 1.

### 1.3 Pourquoi la moyenne échoue et le maximum réussit

Avec un bruit symétrique, la meilleure estimation du niveau d'un patient est une moyenne. Son erreur décroît en 1/√n avec le nombre de mesures n.

Avec un bruit qui ne pousse que vers le bas, la moyenne est biaisée et dispersée par la traîne basse. Le maximum des mesures, lui, s'approche du bord supérieur. Quand la densité du bruit reste forte près de ce bord, l'erreur du maximum décroît en 1/n, bien plus vite. Avec 3 ou 4 mesures par patient, la différence est déjà large.

Un test le confirme. On donne à l'estimateur la forme exacte de la courbe de chaque patient, et il ne doit estimer que son niveau à partir des mesures `off` :

| Estimateur du niveau | RMSE |
|---|---|
| Moyenne des mesures | 4,79 |
| Maximum corrigé du nombre de mesures | 3,76 |

C'est pour cette raison que les variables d'enveloppe de la section 4.4 apportent le plus gros gain mesuré, environ 0,7 point.

---

## 2. Principes directeurs

**P1. Valider par patient.** Le test contient des patients inédits. Une validation par lignes surestime le score.

**P2. Utiliser toutes les visites du patient comme contexte.** `X_test` contient toutes les visites de chaque patient de test. Des résumés calculés à partir de ses seules colonnes sont légitimes, car ils n'utilisent aucune cible du test.

**P3. Apprendre sur le pli d'entraînement tout ce qui utilise la cible.** C'est le cas de l'échelle des évaluateurs. La courbe d'effet du médicament et la sensibilité des patients n'utilisent que `on`, `off` et le délai : elles ne fuient rien.

**P4. Exploiter l'enveloppe haute, pas la moyenne.** C'est la conséquence directe du bruit à sens unique, voir 1.3.

**P5. Convertir `on` par rapport à `off`.** Une fois converti, `on` devient presque aussi informatif que `off`. Cela compte, car 42 % des visites n'ont que `on`.

**P6. Ne pas compter deux fois une visite.** `on` et `off` d'une même visite partagent leur bruit. Une variable qui combine les deux doit prendre l'une ou l'autre, pas les deux comme mesures indépendantes.

**P7. Rapprocher la courbure de la moyenne plutôt que l'ignorer.** Sans régularisation, une droite bat la parabole. Avec un a priori sur la courbure, la parabole gagne : 3,75 contre 3,67 dans le modèle de vraisemblance.

**P8. Lire le classement public comme un signal.** Sur des sous-ensembles de 300 à 500 patients, la RMSE ne varie que de 0,14 à 0,19. Un écart de plus d'environ 0,4 entre validation et classement indique donc une erreur de structure, pas de la malchance.

**P9. Ne pas investir dans les variables démographiques ni dans la dose.** Elles expliquent très peu de la courbe.

---

## 3. Protocole de validation

- **`GroupKFold` à 5 plis** sur `patient_id`. Une stratification n'est pas nécessaire : les plis concordent à 0,1 point près.
- **Calculer les plis une fois, les sauvegarder, les réutiliser.** Les comparaisons deviennent appariées.
- skore appelle `split(X, y)` sans `groups` : passer la liste des paires d'indices précalculées, comme dans le guide du jour 2.
- **Critère de décision :** un changement est retenu s'il améliore au moins 4 plis sur 5 et si le gain moyen dépasse l'écart-type des différences entre plis.
- **Segments à suivre :** visites avec et sans `off`, patients à 4 et à 12 visites, première et dernière visite, tranches de cible. L'erreur grandit avec la sévérité.
- **À chaque soumission,** comparer le score public à la validation. Un écart supérieur à environ 0,4 doit déclencher une recherche de structure manquante, voir P8.

---

## 4. Architecture recommandée

Tout est emballé dans **un seul estimateur scikit-learn**, une classe héritant de `BaseEstimator` et `RegressorMixin`. `X` contient `patient_id` et `age`, utilisés en interne. Ainsi :

- `skore.evaluate` l'accepte avec les plis groupés ;
- tout ce qui utilise la cible est appris dans `fit` ;
- `predict` reçoit des patients complets en validation comme en test.

Seul scikit-learn est nécessaire.

### 4.1 Bloc 1 : variables de visite et de traitement, sans cible

| Variable | Raison |
|---|---|
| `dur` = `age` − `age_at_diagnosis` | Corrélation de 0,54 avec la cible |
| `t_rel` = `age` − âge à la première visite | Disponible même sans âge au diagnostic |
| Rang de visite, nombre de visites, durée du suivi | Situent la visite dans la trajectoire |
| Indicateurs de manque | Le manque suit le stade de la maladie |
| `treat` = `on`, `ledd` ou délai de prise présent | Statut traité à la visite |
| Âge au premier traitement, temps depuis le début du traitement | Le traitement commence au-delà d'une vraie valeur de 18 |
| Âge à la dernière visite non traitée, durée de maladie au début du traitement, nombre de visites traitées | Encadrent le moment où la courbe a franchi le seuil |
| `rater_id` en catégorie | 50 niveaux, tous vus dans le train |

Sans aucune valeur `on` ni `off`, ces seules variables donnent une RMSE de 9,42, contre 14,30 avec les données démographiques et `dur`.

### 4.2 Bloc 2 : conversion de `on`, sans cible

1. **Courbe d'effet du médicament.** Sur les visites qui ont à la fois `on` et `off`, avec `off` d'au moins 8 et `on` d'au moins 1, calculer la médiane du log de `on` sur `off` par tranche de délai depuis la prise. Les tranches sont plus fines au début, où l'effet monte vite. Les visites sans délai connu forment leur propre tranche, car leur médiane diffère, à −0,51 contre −0,58 au plateau.
2. **Sensibilité du patient.** Pour chaque patient, faire la moyenne de ses écarts à cette courbe, rapprochée de zéro : somme des écarts divisée par leur nombre plus 1. Un patient sans visite double reçoit 0.
3. **`on_eq`** = `on` × exp(−(effet de la tranche + sensibilité)). C'est l'équivalent `off` de la mesure `on`.

La courbe peut être apprise dans `fit` sur les `X` d'entraînement. La sensibilité se calcule par patient dans `transform`, à partir de ses seules colonnes.

### 4.3 Bloc 3 : mise à l'échelle par évaluateur, avec la cible

Dans `fit`, pour chaque évaluateur, calculer le 95e centile du rapport `off` sur cible, sur les visites où la cible dépasse 15. Ce centile estime le bord supérieur du bruit de l'évaluateur. On en déduit :

- `u_off` = `off` / centile de l'évaluateur ;
- `u_on` = `on_eq` / centile de l'évaluateur ;
- `u_any` = `u_off` s'il existe, sinon `u_on`. Cette série respecte P6.

### 4.4 Bloc 4 : variables d'enveloppe, sans cible

Pour chaque visite *v* d'un patient, et pour chacune des trois séries `u_off`, `u_on` et `u_any` :

1. Prendre toutes les mesures *j* de ce patient dans la série.
2. Ramener chaque mesure à l'âge de la visite *v* selon une pente fixe *s* : `u_j + s × (âge_v − âge_j)`.
3. Trier ces valeurs ramenées de la plus haute à la plus basse et garder les rangs 1 à 6.

Avec les pentes 0 ; 1,2 ; 2,4 ; 4 et 6 points par an, on obtient 3 × 5 × 6 = 90 variables. Le boosting choisit lui-même quelle pente et quel rang conviennent à chaque situation.

| Variante | RMSE |
|---|---|
| Sans enveloppe | 3,63 |
| 7 variables : rangs 1 et 2, pente 2,4 | 3,22 |
| 36 variables : rangs 1 à 4, trois pentes | 3,01 |
| 90 variables : rangs 1 à 6, cinq pentes | 2,95 |

Ce calcul est du pandas et du numpy simple, par patient. Il se porte facilement dans Bob comme transformeur personnalisé.

On garde aussi les résumés de la version 1 : droite pondérée par patient, médianes, mesures de la visite précédente et suivante. Ils faisaient partie du pipeline à 3,81.

### 4.5 Bloc 5 : gradient boosting

`HistGradientBoostingRegressor`, perte quadratique, puisque la métrique est la RMSE.

- **Point de départ :** `learning_rate` de 0,05, 600 itérations, `max_leaf_nodes` de 31, `min_samples_leaf` de 40, `l2_regularization` de 1.
- **Arrêt anticipé désactivé.** Au-delà de 10 000 lignes, il s'active par défaut et réserve 10 % de lignes tirées au hasard. Ces lignes appartiennent aux mêmes patients que l'entraînement, ce qui fausse l'arrêt. Choisir le nombre d'itérations par la validation groupée.

### 4.6 Bloc 6 : post-traitement par patient

1. **Projection parabolique.** Pour chaque patient, ajuster une parabole de l'âge sur ses prédictions et la substituer. Gain mesuré de 2,95 à 2,89. Cette étape n'a pas de garantie théorique, car la vraie courbe n'est pas exactement une parabole. Elle s'en écarte toutefois de moins d'un tiers de point.
2. **Projection isotonique**, avec `IsotonicRegression` sur les prédictions triées par âge.
3. **Bornes** entre 0 et 132.

Les étapes 2 et 3 sont garanties sans risque. Les suites croissantes forment un ensemble convexe, les suites bornées aussi, et la vraie cible appartient aux deux. Projeter sur un ensemble convexe rapproche de tout point de cet ensemble, donc l'erreur de chaque patient ne peut que baisser ou rester égale. Leur gain mesuré est faible, de 0,01 à 0,02, mais il est gratuit.

### 4.7 Bloc 7, optionnel : modèle de vraisemblance empilé

Un modèle bayésien par patient exprime directement le modèle de mesure de 1.2 :

- **Grille** sur le niveau et la pente de la courbe, avec une courbure fixée à sa médiane de population.
- **Vraisemblance empirique à sens unique :** la densité du rapport `off` sur échelle × T est estimée par histogramme lissé, par tranche de T, sur le pli d'entraînement. De même pour `on_eq` sur les visites sans `off`.
- **Vraisemblance du début de traitement :** probabilité de commencer le traitement selon T et la durée de maladie.
- **A priori :** niveau selon la durée de maladie, pente selon son histogramme dans la population.
- **Sortie :** la moyenne a posteriori de T à chaque visite.

Seul, ce modèle obtient 3,67. Ajouté comme variable au boosting du bloc 5, avec prédictions hors pli internes, il fait passer l'ensemble de 2,89 à 2,85. Chaque validation coûte environ 25 minutes. À réserver au cas où le temps le permet.

---

## 5. Feuille de route des expériences

Chaque expérience produit un rapport skore poussé sur le Hub avec une clé nouvelle. Chaque étape ajoute un seul bloc, pour mesurer son effet isolé. Les valeurs attendues viennent des essais exploratoires sur les mêmes plis.

| Clé | Changement | RMSE attendue | Priorité |
|---|---|---|---|
| `02_ridge` | Ridge, première soumission Kaggle, exigée par le guide | | Exigée |
| `03_hgb_groupcv` | Boosting sur colonnes brutes, plis groupés | 7,8 | Indispensable |
| `04_patient_aggregates` | Bloc 1 et résumés par patient | 4,4 | Indispensable |
| `05_on_eq_treatment` | Blocs 2 et 3, variables de traitement, corrections apprises | 3,6 | Indispensable |
| `06_envelope` | Bloc 4 | 2,95 | **Le plus gros gain** |
| `07_postproc` | Bloc 6 | 2,89 | Indispensable, coût minime |
| `08_likelihood_stack` | Bloc 7 | 2,85 | Si le temps le permet |
| `09_ensemble` | Graines multiples, réglage limité | Petit gain | Si le temps le permet |
| `10_final` | Ajustement sur tout le train, soumission | | Indispensable |

**Version minimale si le temps manque :** `04`, puis l'enveloppe du bloc 4 sur `u_off` seul, puis le bloc 6. Cette combinaison exacte n'a pas été mesurée. Elle repose sur les blocs dont le gain mesuré est le plus fort.

**Soumissions Kaggle :** `02_ridge` pour valider la chaîne, puis après `04`, après `07`, et la finale. À chaque fois, comparer avec la validation, voir P8.

---

## 6. Ce qu'on ne fait pas, et pourquoi

| Idée | Raison de l'écarter |
|---|---|
| Validation par lignes | Fuite des visites d'un patient, scores trop optimistes |
| Moyenne ou régression des mesures par patient comme estimation principale | Inadaptée au bruit à sens unique. Le meilleur regroupement gaussien obtient 4,8 à 5,0 |
| Corriger `off` du délai depuis la dernière prise | La sous-estimation existe sans traitement. Le délai `off` ne change presque rien |
| Corriger `on` par rapport à la cible | Mélange le bruit de visite et l'effet du médicament. La conversion via `off` est bien plus précise |
| Retirer les mesures très basses comme aberrantes | Elles font partie du bruit attendu. L'enveloppe les ignore naturellement |
| Ingénierie sur `gene`, `sexM`, `cohort`, ou forme du bruit selon ces groupes | Le bruit relatif vaut 0,27 à 0,29 dans chaque groupe |
| Dose comme mesure de sévérité | Elle suit le temps |
| Transformer la cible | Distribution symétrique, métrique RMSE |
| Réglage intensif des hyperparamètres en premier | Les gains viennent de la structure, pas du réglage |
| Utiliser `patient_id` comme variable | Aucun patient commun avec le test |

---

## 7. Risques et parades

| Risque | Symptôme | Parade |
|---|---|---|
| Échelle des évaluateurs apprise sur tout le train | Validation trop optimiste | Centiles appris dans `fit` uniquement |
| Variables d'enveloppe calculées à travers plusieurs patients | Validation anormalement basse | Calcul strictement par `patient_id`, test unitaire sur un faux patient |
| Une visite comptée deux fois dans `u_any` | Gain plus faible qu'attendu | `u_off` en priorité, `u_on` seulement si `off` manque |
| Arrêt anticipé du boosting sur lignes aléatoires | Surapprentissage | `early_stopping=False` |
| Ordre des visites perdu avant le post-traitement | Dégradation après le bloc 6 | Trier par patient puis âge, puis restaurer l'ordre d'origine |
| Désalignement de la soumission | Score proche de la moyenne constante | Vérifier que l'index de soumission égale celui de `X_test` |
| Temps de calcul du bloc 7 | Validation de 25 minutes | Le garder pour la fin, avec des plis déjà figés |
| Soumission invalide | Ligne refusée | Code exécuté dans Bob, rapport skore poussé, URL dans la description, clé nouvelle |

---

## 8. Organisation d'équipe suggérée

Pour une équipe de quatre :

- **Personne 1 :** plis sauvegardés, estimateur enveloppe, intégration skore, soumissions.
- **Personne 2 :** blocs 1 et 2, variables de traitement et conversion de `on`.
- **Personne 3 :** blocs 3 et 4, échelle des évaluateurs et variables d'enveloppe.
- **Personne 4 :** bloc 6, analyse d'erreur par segment, puis bloc 7 s'il reste du temps.

---

## 9. Pistes pour descendre sous 2,85

Ces pistes sont des hypothèses non vérifiées. Elles vont dans la même direction que les gains déjà mesurés :

1. **Estimer la sensibilité au médicament dans le modèle de vraisemblance**, au lieu de l'y injecter comme valeur fixe.
2. **Exploiter le petit bruit propre à `on`** dans les visites qui ont les deux mesures. Une fois le bruit commun retiré, `on` apporte une information indépendante.
3. **Affiner la densité du bruit selon T**, avec un terme additif pour les cibles basses, où l'arrondi à l'entier pèse lourd.
4. **Un modèle au niveau du patient** qui reçoit l'ensemble trié de ses mesures, plutôt qu'un modèle visite par visite.

On ne peut pas exclure que l'équipe en tête dispose d'une information non trouvée ici. Les vérifications faites n'ont révélé aucune fuite par les identifiants, l'ordre des lignes, les doublons de patients ou un décalage entre train et test.

---

## 10. Ce qui a changé depuis la version 1

| Affirmation de la version 1 | Statut |
|---|---|
| Bruit de mesure symétrique, regroupement par moyenne pondérée | **Faux.** Le bruit est à sens unique |
| Plancher atteignable d'environ 3 | **Faux.** 2,85 est déjà atteint |
| Classement public surtout bruité | **Faux.** Il ne varie que de 0,14 à 0,19 selon les patients |
| `on` beaucoup plus bruité que `off`, à faible poids | **Faux** une fois `on` converti par rapport à `off` |
| Le médicament résiduel explique la sous-estimation de `off` | **Faux.** Elle existe sans traitement |
| Une droite bat la parabole | **Vrai seulement sans régularisation** |
| Version minimale : résumés, droite par patient, isotonique | **Mauvaise priorité.** La droite seule fait 6,10 |
| Stratifier les plis | **Inutile** |
| Valider par patient, apprendre dans le pli, désactiver l'arrêt anticipé, empiler hors pli | **Confirmé** |
| La projection isotonique ne peut pas dégrader l'erreur | **Confirmé**, gain faible |
| Variables démographiques et dose peu utiles | **Confirmé** |

---

## 11. Implémentation de référence

Toutes les recommandations de la relecture sont appliquées dans `scratch/pipeline/true_off.py`, sous la forme d'un seul estimateur `TrueOffRegressor`. Il est compatible avec scikit-learn et avec `skore.evaluate`. Chaque bloc de la section 4 y est une fonction distincte, pour faciliter le portage dans Bob.

| Variante | RMSE en validation groupée |
|---|---|
| Sans variables d'enveloppe | 3,538 |
| Sans projection parabolique | 2,955 |
| Pipeline par défaut | 2,898 |
| Avec modèle bayésien empilé, 15 minutes par validation | **2,839** |

Les variantes isolent l'apport de chaque bloc : environ 0,64 point pour l'enveloppe, 0,06 pour la projection parabolique et 0,06 pour le modèle bayésien.

Quatre tests vérifient les garde-fous de la section 7 :

- la prédiction d'un patient ne dépend pas des autres patients présents ;
- l'ordre des lignes ne change pas les prédictions ;
- les prédictions sont croissantes par patient et bornées entre 0 et 132 ;
- la prédiction sur `X_test` a la forme attendue par `sample_submission.csv`.
