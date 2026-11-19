# RAPPORT FINAL DE QUALIFICATION — AI-TestGuard

**Projet :** Validation, robustesse et fiabilité d’un modèle de détection de fraude bancaire  
**Formation :** Master 2 Data Science et Intelligence Artificielle (MDSIA 2)  
**Auteur :** Gibril KAMADEU  
**Date du rapport :** 19 novembre 2026  
**Dépôt :** https://github.com/kagibru/AI-TestGuard

## 1. Le contexte et le périmètre de l’évaluation.

AI-TestGuard vise à qualifier un pipeline de détection de transactions frauduleuses par carte bancaire. La qualification porte sur la validité des entrées, la fiabilité des prédictions, l’évaluation statistique, l’intégration, la robustesse, la non-régression, les performances techniques et l’automatisation des contrôles.

Le jeu de données **Credit Card Fraud Detection — Kaggle** contient **284 807 transactions**, dont **492 fraudes** (environ **0,173 %**). La forte rareté de la classe positive rend insuffisante une évaluation fondée uniquement sur l’*accuracy*. Les faux positifs peuvent perturber des transactions légitimes ; les faux négatifs exposent la banque à des fraudes non détectées.

Le périmètre testé concerne le pipeline Python/scikit-learn et ses artefacts. Aucun service API HTTP de production, dispositif d’authentification bancaire ni environnement d’exploitation réel n’a été qualifié.

## 2. Le plan de test et la matrice de traçabilité.

La démarche est fondée sur les risques et sur une matrice reliant chaque exigence aux tests et aux preuves. Les identifiants **T01 à T07** sont ceux de l’énoncé ; **T08 à T12** ont été ajoutés dans le projet et étendent la couverture sans renuméroter les exigences existantes.

| Exigences | Objet contrôlé | Preuves / statut |
|---|---|---|
| T01–T03 | Schéma, types numériques, sorties finies | `tests/test_preprocessing.py` — contrôles exécutés |
| T04 | Probabilités de prédiction valides | `tests/test_prediction.py` |
| T05 | Indépendance du jeu de test et intégration | `tests/test_integration.py` ; test Kaggle local |
| T06 | Reproductibilité | `tests/test_regression.py` |
| T07 | Automatisation et signalisation CI | GitHub Actions : run #4 réussi, 19/11/2026 |
| T08–T09 | Cible binaire ; persistance du pipeline | Tests de validation et de persistance |
| T10–T11 | Baseline et robustesse perturbée | `reports/baseline_model.json` ; `reports/robustness_results.json` |
| T12 | Performance et contrôles de sécurité | `tests/test_performance_security.py` ; `reports/benchmark_model.json` |

## 3. L’architecture du pipeline et les choix méthodologiques.

Le pipeline sépare la validation et le prétraitement (`app/preprocessing.py`), l’entraînement (`app/training.py`), la prédiction (`app/prediction.py`), l’évaluation (`app/evaluation.py`) et la persistance (`app/persistence.py`). La chaîne retenue associe `StandardScaler` à `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.

Un `DecisionTreeClassifier` sert de modèle comparatif.

| Étape | Méthode | Justification |
|---|---|---|
| Données | 284 807 lignes ; cible `Class` binaire | Jeu public déséquilibré |
| Séparation | 60 % train / 20 % validation / 20 % test ; stratification ; seed 42 | Limiter les fuites et stabiliser les comparaisons |
| Apprentissage | Prétraitement ajusté uniquement sur train | Éviter la fuite statistique |
| Décision | Seuil 0,90 retenu sur validation | Compromis précision/rappel documenté |
| Persistance | `joblib` ; rechargement et comparaison des prédictions | Vérifier la cohérence de l’artefact |

**Répartition observée :** entraînement **170 884** transactions (**295 fraudes**), validation **56 961** (**98 fraudes**), test **56 962** (**99 fraudes**). Le jeu de test indépendant a été utilisé pour l’évaluation finale, sans réajustement ultérieur du modèle ou du seuil.

## 4. Les résultats des tests unitaires et d’intégration.

La campagne GitHub Actions du 19 novembre 2026 a finalement abouti à une exécution réussie sur la branche `main`, après correction des dépendances Linux et exclusion du test Kaggle de la CI. La campagne précédente exécutait **56 tests**, dont **55 réussis** et **un échec** lié à l’absence de `data/creditcard.csv` sur le runner. Sa couverture combinée était de **96,39 %**.

| Contrôle | Résultat documenté | Interprétation |
|---|---|---|
| Tests automatisés (run précédent) | 55 réussis / 56 ; 1 erreur de fichier Kaggle | Échec d’environnement, non de prédiction |
| Couverture (run précédent) | 96,39 % | Couverture combinée mesurée lors de ce run ; preuve : `reports/coverage.xml` si conservée |
| CI après correction | Run #4 réussi, commit `44f6389` | Chaîne de contrôles configurés réussie |
| Tests d’intégration Kaggle | Exécution locale distincte | `reports/integration_tests.xml` si le rapport a été généré et conservé |
| Tests T12 | Trois contrôles implémentés | `tests/test_performance_security.py` ; résultat Pytest individuel à vérifier |

> **Limite de preuve :** le succès du run #4 ne prouve pas à lui seul l’exécution du test Kaggle exclu de la CI. Les résultats locaux et le statut exact des trois tests T12 doivent être appuyés par leurs sorties d’exécution.

## 5. Les métriques de performance et les matrices de confusion.

La validation a comparé la régression logistique à un arbre de décision. Les métriques ci-dessous proviennent du **jeu de validation**, au seuil **0,90**, et sont distinctes des résultats du jeu de test final.

| Modèle / validation | Précision | Rappel | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0,2516 | 0,8265 | 0,3857 | 0,9806 | 0,6857 |
| DecisionTreeClassifier | 0,2057 | 0,8163 | 0,3285 | 0,9124 | 0,6316 |

Pour `LogisticRegression`, la validation relève **241 faux positifs** et **17 faux négatifs**. Sur **98 fraudes**, **81** sont détectées ; sur **56 863 transactions légitimes**, **56 622** sont correctement écartées. Pour l’arbre, les résultats correspondants sont **309 faux positifs** et **18 faux négatifs**.

**Matrice de confusion — validation (régression logistique)**

| Classe réelle | Prédit normal (0) | Prédit fraude (1) |
|---|---:|---:|
| Réel normal (0) | TN = 56 622 | FP = 241 |
| Réel fraude (1) | FN = 17 | TP = 81 |

La régression logistique offre, sur ce jeu de validation, une meilleure précision, un meilleur F1 et une meilleure PR-AUC que l’arbre. Toutefois, une précision de **25,16 %** signifie qu’environ trois alertes sur quatre sont des faux positifs. La capacité opérationnelle à traiter ces alertes doit être évaluée.

### Évaluation finale hors échantillon

L’évaluation finale hors échantillon a été réalisée sur **56 962 transactions**, dont **99 frauduleuses**, en utilisant le modèle de régression logistique sauvegardé et le seuil de décision de **0,90**, préalablement sélectionné sur le jeu de validation.

Le modèle obtient une **précision de 25,73 %**, un **rappel de 88,89 %**, un **F1-score de 39,91 %**, une **ROC-AUC de 96,62 %** et une **PR-AUC de 75,75 %**.

**Matrice de confusion — test final**

| Classe réelle | Prédit normal (0) | Prédit fraude (1) |
|---|---:|---:|
| Réel normal (0) | TN = 56 609 | FP = 254 |
| Réel fraude (1) | FN = 11 | TP = 88 |

**Comparaison des performances prédictives — validation et test final** (équivalent tabulaire du graphique du PDF) :

| Indicateur | Validation | Test final |
|---|---:|---:|
| Précision | 25,16 % | 25,73 % |
| Rappel | 82,65 % | 88,89 % |
| F1-score | 38,57 % | 39,91 % |
| ROC-AUC | 98,06 % | 96,62 % |
| PR-AUC | 68,57 % | 75,75 % |

Ces résultats confirment une capacité élevée à identifier les transactions frauduleuses sur le jeu de test indépendant. La proportion importante de faux positifs demeure toutefois une limite opérationnelle majeure.

**Preuve :** `reports/final_test_evaluation.json`.

## 6. Les tests de robustesse et de non-régression.

Les tests de robustesse vérifient la capacité du modèle à rejeter les données invalides et mesurent sa sensibilité à des perturbations contrôlées. Les résultats suivants concernent le jeu de validation, au seuil 0,90.

| Scénario | Précision | Rappel | F1 | FP / FN |
|---|---:|---:|---:|---:|
| Référence | 0,2516 | 0,8265 | 0,3857 | 241 / 17 |
| Bruit 1 % écart-type | 0,2508 | 0,8367 | 0,3859 | 245 / 16 |
| `Amount` +20 % | 0,2152 | 0,8367 | 0,3424 | 299 / 16 |

L’augmentation de 20 % du montant produit **58 faux positifs supplémentaires** et une baisse du F1 d’environ **0,0433** : cette sensibilité doit être considérée dans les limites du modèle. Les tests de non-régression T10 comparent les résultats à une baseline figée, avec des tolérances explicites sur la précision, le rappel, le F1, les AUC et les erreurs FP/FN.

La reproductibilité est recherchée par une graine aléatoire fixe, une séparation stable et des paramètres de modèle documentés. Le contrôle basé sur un rapport de métriques sauvegardé doit être complété par une régénération des métriques lors des changements de modèle.

## 7. Les mesures de latence et de consommation de ressources.

Le benchmark a comporté **100 itérations** après **10 échauffements**, sur des lots de **100 transactions** du jeu de validation. Il utilise `time.perf_counter`, `psutil` et la taille de l’artefact `joblib`.

| Indicateur | Mesure / protocole | Résultat (`reports/benchmark_model.json`) |
|---|---|---:|
| Latence moyenne du lot | Moyenne de 100 appels `predict_proba` | **0,567 ms** |
| P95 du lot de 100 | 95e percentile des temps par lot | **0,614 ms** |
| Débit | 10 000 transactions / temps cumulé | **176 244 transactions/s** |
| Taille du modèle | Taille de `fraud_logistic_regression.joblib` | **2,66 Ko** |
| Mémoire | Variation RSS du processus avant/après | **0 Mo** |

Le benchmark mesure l’inférence en lots, non la latence d’une API HTTP ou d’une transaction isolée. Une variation nette de RSS de 0 Mo ne signifie pas une consommation mémoire nulle. Les seuils de service doivent être définis en fonction d’un contexte métier et d’une infrastructure représentative.

## 8. Les anomalies, leur gravité et leur statut.

| Anomalie / risque | Gravité | Statut / mesure |
|---|---|---|
| Dépendance Windows `pywinpty` non compilable sous Linux | Majeure | Corrigée : `requirements.txt` épuré ; la CI installe les dépendances |
| Test T05 dépendant du fichier Kaggle absent en CI | Majeure | Contournée : test marqué `integration` exclu du run CI ; exécution locale distincte |
| Trois exécutions GitHub Actions en échec | Mineure après correction | Historique conservé ; run #4 réussi |
| Faux positifs élevés en validation et au test final | Majeure métier | Ouverte : analyse du coût des alertes et calibration |
| Sensibilité à `Amount` +20 % | Modérée | Ouverte : investigation de robustesse |
| Sécurité des artefacts `joblib` | Majeure potentielle | Charger uniquement des artefacts fiables ; contrôler leur provenance |
| Évaluation finale hors échantillon initialement non documentée | Majeure initialement | **Résolue sur le plan documentaire** : `reports/final_test_evaluation.json` généré |

Les exécutions CI échouées attestent la détection d’incidents techniques ; elles ne remplacent pas nécessairement la preuve exigée d’un test volontairement défaillant.

## 9. Les limites de l’évaluation et les risques résiduels.

Les variables `V1` à `V28` sont anonymisées. Le jeu ne fournit pas les caractéristiques démographiques nécessaires pour mesurer les disparités de performance selon des groupes protégés : aucune conclusion d’équité ne peut être établie.

L’échantillon ne garantit ni représentativité temporelle ni transférabilité à d’autres banques ou circuits de paiement. La surveillance des dérives, des taux d’alerte, de la latence, de la disponibilité et des erreurs FP/FN est décrite dans `reports/monitoring_plan.md` (sous réserve de présence de ce fichier dans le dépôt), mais n’a pas été démontrée dans une infrastructure de production. Les labels de fraude peuvent être retardés ; les métriques de performance réelles ne sont donc pas immédiatement observables.

Les principaux risques résiduels sont les faux positifs, les fraudes non détectées, le *data drift* et le *concept drift*, les accès non autorisés aux modèles, l’absence de supervision opérationnelle éprouvée et l’incertitude sur la performance future.

## 10. Une décision argumentée sur l’aptitude du modèle à poursuivre vers une phase pilote.

L’évaluation finale sur le jeu de test indépendant confirme des performances prédictives encourageantes, avec un **rappel de 88,89 %** et une **PR-AUC de 75,75 %**.

Les résultats des tests logiciels, des contrôles de robustesse et de reproductibilité, ainsi que des mesures techniques, constituent des éléments favorables à la poursuite du projet.

Toutefois, la **précision de 25,73 %** révèle une proportion importante de fausses alertes, susceptible de générer une charge opérationnelle significative.

En conséquence, le modèle est considéré comme **apte à poursuivre vers une phase pilote encadrée, sous conditions**, notamment la validation des critères métier, l’évaluation du coût des faux positifs et des fraudes non détectées, la mise en place d’un monitoring des performances et la réalisation de tests dans un environnement représentatif. **Cette décision ne vaut pas autorisation de mise en production.**

## Références et pièces justificatives

Les éléments de preuve de la qualification du modèle AI-TestGuard sont constitués des scripts Python, des tests automatisés, des rapports d’évaluation et des résultats d’intégration continue. Ils permettent d’assurer la traçabilité des contrôles réalisés et de justifier les conclusions du présent rapport.

### Références méthodologiques et techniques

- **Scikit-learn** : bibliothèque utilisée pour le prétraitement, l’entraînement, la prédiction et l’évaluation des modèles.
- **Pytest et pytest-cov** : outils d’exécution des tests automatisés et de mesure de leur couverture.
- **GitHub Actions** : plateforme d’intégration continue utilisée pour automatiser les contrôles de qualité du code et les tests.
- **Credit Card Fraud Detection — Kaggle** : jeu de données de référence comprenant 284 807 transactions, dont 492 frauduleuses. Source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

### Pièces justificatives du projet

| Référence | Fichier ou élément de preuve | Objet |
|---|---|---|
| PJ-01 | `reports/plan_de_test.md` | Plan de test, exigences T01 à T12 et matrice de traçabilité |
| PJ-02 | `tests/` | Tests unitaires, d’intégration, de robustesse, de non-régression et de sécurité |
| PJ-03 | `reports/validation_logistic_regression.json` | Résultats de validation du modèle de régression logistique |
| PJ-04 | `reports/model_comparison.json` | Comparaison des modèles de classification |
| PJ-05 | `reports/robustness_results.json` | Résultats des scénarios de robustesse |
| PJ-06 | `reports/baseline_model.json` | Référence utilisée pour les contrôles de non-régression |
| PJ-07 | `reports/analyse_biais_limites.md` | Analyse des biais potentiels et des limites méthodologiques |
| PJ-08 | `reports/benchmark_model.json` | Mesures de latence, de débit, de taille du modèle et de mémoire |
| PJ-09 | `scripts/benchmark_model.py` | Script de mesure des performances techniques |
| PJ-10 | `reports/final_test_evaluation.json` | Évaluation finale indépendante sur 56 962 transactions, dont 99 frauduleuses |
| PJ-11 | `scripts/evaluate_final_test.py` | Script de reproduction de l’évaluation finale avec le seuil fixé à 0,90 |
| PJ-12 | `.github/workflows/tests.yml` | Configuration du pipeline d’intégration continue |
| PJ-13 | GitHub Actions — exécution #4 du 19/11/2026 | Preuve de réussite du workflow d’intégration continue |
| PJ-14 | `README.md` | Présentation du projet, instructions d’installation et modalités d’exécution |

### Résultats de référence retenus pour la qualification

- **Évaluation finale indépendante :** précision de **25,73 %**, rappel de **88,89 %**, F1-score de **39,91 %**, ROC-AUC de **96,62 %** et PR-AUC de **75,75 %**.
- **Matrice de confusion finale :** **56 609** vrais négatifs, **254** faux positifs, **11** faux négatifs et **88** vrais positifs.
- **Benchmark technique :** latence moyenne de **0,567 ms** par lot de 100 transactions, latence P95 de **0,614 ms**, débit de **176 244 transactions/s** et taille du modèle d’environ **2,66 Ko**.
- **Intégration continue :** exécution réussie du workflow GitHub Actions #4, avec un seuil minimal de couverture configuré à **85 %**.

### Traçabilité et conservation des preuves

Les scripts et rapports destinés à la qualification sont conservés dans le dépôt GitHub du projet : https://github.com/kagibru/AI-TestGuard

Les fichiers de données Kaggle et le modèle entraîné sont conservés localement et exclus du versionnement Git. Les tests qui dépendent de ces fichiers nécessitent donc un environnement local disposant des données et des artefacts appropriés.

