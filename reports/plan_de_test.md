
# AI-TestGuard — Plan de test


## 1. Contexte et objectifs

AI-TestGuard est un projet de qualification d'un système
de Machine Learning destiné à détecter les fraudes
sur les transactions par carte bancaire.

L'objectif n'est pas uniquement d'obtenir de bonnes
performances prédictives, mais de démontrer que le
pipeline est correctement testé, robuste, reproductible
et exploitable dans des conditions documentées.

Deux risques métier majeurs sont identifiés :
- Faux positifs : transactions légitimes signalées comme frauduleuses.
- Faux négatifs : transactions frauduleuses non détectées.

Le projet s'appuie sur le jeu de données
Credit Card Fraud Detection disponible sur Kaggle.

Le jeu original contient 284 807 transactions,
dont 492 frauduleuses, soit environ 0,173 %.

## 2. Périmètre du système

### 2.1 Composants inclus

- Chargement et validation du jeu de données.
- Contrôle du schéma, des types et des valeurs.
- Prétraitement des variables explicatives.
- Partitionnement entraînement/validation/test.
- Entraînement des modèles de classification.
- Production de probabilités et de prédictions.
- Évaluation des performances prédictives.
- Sauvegarde et rechargement du modèle.
- Tests d'intégration et de robustesse.
- Tests de non-régression et de reproductibilité.
- Mesure des performances techniques.
- Automatisation des contrôles avec GitHub Actions.

### 2.2 Limites du périmètre

- Pas d'intégration réelle à un système bancaire.
- Pas de traitement de transactions en production.
- Pas de démonstration de l'équité démographique,
  les attributs nécessaires étant absents.
- Pas de garantie de généralisation aux transactions
  bancaires actuelles.

### 2.3 Précautions méthodologiques

- Ne pas utiliser l'accuracy comme seule métrique.
- Séparer entraînement, validation et test avant
  l'ajustement des transformations apprises.
- Ne pas utiliser le test final pour sélectionner
  le modèle, les hyperparamètres ou le seuil.
- Préserver la proportion des classes lorsque
  la stratification est appropriée.
- Fixer les graines aléatoires.
- Ne pas attribuer de signification métier directe
  aux composantes anonymisées V1 à V28.

## 3. Exigences fonctionnelles

Les exigences fonctionnelles définissent les comportements attendus du pipeline de détection de fraude bancaire. Chaque exigence possède un identifiant unique conforme à la nomenclature du sujet.

### T01 — Présence des colonnes nécessaires

Le système doit vérifier que toutes les colonnes indispensables sont présentes dans le jeu de données : `Time`, `V1` à `V28`, `Amount` et `Class`.

**Résultat attendu :** aucune colonne indispensable manquante. En cas d'absence, le système doit signaler explicitement l'anomalie.

### T02 — Conformité des types de données

Le système doit contrôler que les variables respectent les types définis dans le contrat de données.

**Résultat attendu :** types conformes au contrat. Toute valeur incompatible doit être rejetée ou traitée selon une règle documentée.

### T03 — Validité des données après prétraitement

Le prétraitement doit produire des données numériques exploitables par les modèles de Machine Learning.

**Résultat attendu :** aucune valeur `NaN` ou infinie en sortie du prétraitement. Les dimensions des données doivent également être conformes aux attentes.

### T04 — Production de probabilités valides

Le modèle doit produire une probabilité estimée de fraude pour chaque transaction valide.

**Résultat attendu :** chaque probabilité est finie et comprise entre 0 et 1.

### T08 — Validité de la variable cible

Le système doit vérifier que la variable `Class` contient exclusivement les valeurs autorisées : `0` (transaction normale) et `1` (transaction frauduleuse).

**Résultat attendu :** toute valeur différente de 0 ou 1 est détectée et rejetée.

### T09 — Conservation des prédictions après rechargement

Le système doit permettre la sauvegarde et le rechargement du pipeline entraîné, comprenant les transformations nécessaires et le modèle de classification.

**Résultat attendu :** pour les mêmes données d'entrée, le pipeline rechargé produit les mêmes prédictions et probabilités que le pipeline initial, dans la tolérance numérique définie.

### T11 — Gestion des entrées invalides et perturbées

Le système doit gérer les situations anormales prévues dans la stratégie de test : valeurs manquantes, valeurs infinies, types incorrects, montants extrêmes et perturbations numériques.

**Résultat attendu :** les entrées invalides sont rejetées explicitement ou corrigées selon une règle documentée. Les effets des perturbations admissibles sur les prédictions sont mesurés.

## 4. Exigences non fonctionnelles

Les exigences non fonctionnelles définissent les propriétés de qualité du pipeline : indépendance de l'évaluation, reproductibilité, fiabilité, performances techniques et automatisation des contrôles.

### T05 — Indépendance des données d'évaluation

Les performances du modèle doivent être évaluées sur des données indépendantes de celles utilisées pour son entraînement et son réglage.

**Résultat attendu :** le jeu de test final n'est utilisé ni pour entraîner le modèle, ni pour sélectionner ses hyperparamètres, ni pour choisir son seuil de décision.

### T06 — Reproductibilité des résultats

Les expériences doivent être reproductibles dans des conditions documentées : versions logicielles, paramètres, graines aléatoires et jeux de données.

**Résultat attendu :** les résultats reproduits présentent des écarts compatibles avec les tolérances définies.

### T07 — Automatisation des tests avec GitHub Actions

Les contrôles essentiels doivent pouvoir être exécutés automatiquement dans un environnement d'intégration continue.

**Résultat attendu :** le workflow GitHub Actions réussit lorsque les contrôles passent et signale correctement un échec lorsqu'un test échoue.

### T10 — Détection des régressions de performance

Le système de qualification doit comparer les performances d'une nouvelle version du modèle à une référence enregistrée (*baseline*).

**Résultat attendu :** toute dégradation dépassant les tolérances définies est détectée et signalée.

### T12 — Mesure des performances techniques

Le système doit permettre de mesurer les performances techniques de l'inférence et les ressources nécessaires à son exécution.

**Résultat attendu :** les mesures de latence, notamment le P95, de débit, de taille du modèle et de consommation mémoire sont documentées selon un protocole reproductible.

---

**Note de traçabilité :** les exigences T01 à T07 conservent les identifiants et le sens de la matrice officielle du sujet. Les exigences T08 à T12 constituent les cinq exigences complémentaires proposées dans le cadre d'AI-TestGuard. Les risques associés, les cas de test et les preuves seront détaillés dans les sections suivantes du plan de test.

## 5. Analyse des risques

### 5.1. Objectif et méthode

L'analyse des risques vise à identifier les défaillances susceptibles d'affecter la qualité des données, la fiabilité des prédictions, la reproductibilité des expériences et la qualification du pipeline AI-TestGuard.

Chaque risque est rattaché à une exigence identifiée de T01 à T12.

La criticité initiale est évaluée selon deux critères :

- **Probabilité (P)** : 1 = faible, 2 = moyenne, 3 = élevée.
- **Impact (I)** : 1 = mineur, 2 = significatif, 3 = critique.

**Criticité = P × I**

Interprétation :

- **1 à 2** : criticité faible ;
- **3 à 4** : criticité modérée ;
- **6 à 9** : criticité élevée.

Les évaluations ci-dessous sont des estimations préliminaires. Elles seront révisées à partir des résultats des tests et des anomalies constatées.

### 5.2. Registre des risques

| Exigence | Risque identifié | Conséquence potentielle | P | I | Criticité |
|---|---|---|---:|---:|---:|
| T01 | Colonnes obligatoires absentes ou dupliquées | Erreur de traitement ou incohérence des variables | 2 | 3 | 6 — Élevée |
| T02 | Types de données incompatibles | Échec du traitement ou conversion incorrecte | 2 | 2 | 4 — Modérée |
| T03 | Valeurs NaN ou infinies après prétraitement | Échec de l'entraînement ou prédictions invalides | 2 | 3 | 6 — Élevée |
| T04 | Probabilités invalides ou non finies | Décisions de classification incohérentes | 1 | 3 | 3 — Modérée |
| T05 | Fuite de données entre entraînement, validation et test | Surestimation des performances et conclusions trompeuses | 2 | 3 | 6 — Élevée |
| T06 | Résultats non reproductibles | Comparaisons et vérifications non fiables | 2 | 2 | 4 — Modérée |
| T07 | Tests non exécutés ou échecs non détectés dans GitHub Actions | Régression introduite sans alerte | 2 | 3 | 6 — Élevée |
| T08 | Valeurs de cible autres que 0 et 1 | Apprentissage d'une classification non conforme | 2 | 3 | 6 — Élevée |
| T09 | Pipeline rechargé différent du pipeline initial | Prédictions incohérentes après sérialisation | 1 | 3 | 3 — Modérée |
| T10 | Dégradation des métriques non détectée | Adoption d'une version moins performante | 2 | 3 | 6 — Élevée |
| T11 | Entrées invalides ou perturbées mal gérées | Erreurs, instabilité ou prédictions dégradées | 3 | 2 | 6 — Élevée |
| T12 | Latence excessive ou consommation de ressources non maîtrisée | Dégradation du service ou difficulté d'exploitation | 2 | 2 | 4 — Modérée |

### 5.3. Mesures de maîtrise des risques

| Exigence | Mesure de maîtrise | Preuve attendue |
|---|---|---|
| T01 | Vérification du schéma, des colonnes manquantes et dupliquées | Tests pytest de validation |
| T02 | Contrôle des types et des conversions autorisées | Tests pytest sur types valides et invalides |
| T03 | Contrôle des valeurs finies et des dimensions après transformation | Rapport des tests de prétraitement |
| T04 | Vérification des probabilités produites sur des entrées valides | Tests de prédiction |
| T05 | Séparation entraînement/validation/test et ajustement des transformations uniquement sur l'entraînement | Tests d'intégration et documentation du pipeline |
| T06 | Fixation des graines, paramètres et versions ; comparaison avec une référence | Rapport de reproductibilité |
| T07 | Exécution automatisée de pytest et Ruff avec signalement des échecs | Journaux GitHub Actions |
| T08 | Vérification de l'appartenance des valeurs de `Class` à l'ensemble {0, 1} | Tests sur cibles valides et invalides |
| T09 | Comparaison des prédictions avant et après sauvegarde/rechargement | Test d'intégration de sérialisation |
| T10 | Comparaison des métriques avec une baseline et des tolérances documentées | Rapport de non-régression |
| T11 | Tests d'entrées invalides, de valeurs extrêmes et de perturbations numériques | Rapport de robustesse |
| T12 | Benchmark documenté de latence, débit, taille du modèle et mémoire | Rapport de performance technique |

### 5.4. Priorisation

Les risques de criticité élevée seront traités en priorité, notamment :

- **T05 — Fuite de données** : risque majeur pour la validité scientifique des résultats.
- **T01, T03 et T08 — Qualité des données** : risques susceptibles de compromettre l'ensemble du pipeline.
- **T07 et T10 — Automatisation et non-régression** : risques de défauts non détectés.
- **T11 — Robustesse** : risque d'instabilité face à des entrées invalides ou perturbées.

La réussite des tests ne supprimera pas automatiquement tous les risques. Les risques résiduels seront documentés dans le rapport final de qualification.

### 5.5. Suivi des risques

Pour chaque risque, le suivi reposera sur :

1. L'exécution du ou des tests associés.
2. L'enregistrement des résultats et anomalies.
3. L'analyse des défaillances observées.
4. La mise en œuvre éventuelle d'une correction.
5. La réexécution des tests concernés.
6. L'évaluation du risque résiduel.

Le statut des risques sera actualisé lors de la qualification finale du modèle.

## 6. Matrice de traçabilité

### 6.1. Objectif

La matrice de traçabilité établit la correspondance entre les exigences du projet AI-TestGuard, les risques identifiés, les tests prévus, leurs résultats attendus et les preuves d'exécution.

Les exigences T01 à T07 reprennent la nomenclature officielle du sujet. Les exigences T08 à T12 constituent les cinq exigences complémentaires définies pour le projet.

Cette matrice sera actualisée au cours des activités de développement, de test et de qualification.

### 6.2. Matrice des exigences et des tests

| ID | Exigence | Type de test | Risque maîtrisé | Résultat attendu |
|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Colonnes obligatoires manquantes ou dupliquées | Aucune colonne indispensable manquante |
| T02 | Les données respectent les types attendus | Unitaire | Types incompatibles | Types conformes au contrat |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | Valeurs NaN ou infinies | Aucune valeur NaN ou infinie en sortie |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Probabilités invalides | Probabilité comprise entre 0 et 1 |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Fuite de données | Jeu de test non utilisé pour le réglage |
| T06 | Les résultats sont reproductibles | Non-régression | Résultats non reproductibles | Écart compatible avec la tolérance définie |
| T07 | Les tests s'exécutent sur GitHub | CI | Défauts non détectés | Workflow réussi ou échec correctement signalé |
| T08 | La variable cible contient exclusivement 0 ou 1 | Unitaire | Cible invalide | Toute valeur hors de {0, 1} est rejetée |
| T09 | Le pipeline conserve ses prédictions après rechargement | Intégration | Incohérence après sérialisation | Prédictions identiques dans la tolérance définie |
| T10 | Les régressions de performance sont détectées | Non-régression | Dégradation non détectée | Toute dégradation dépassant les tolérances est signalée |
| T11 | Les entrées invalides ou perturbées sont maîtrisées | Robustesse | Instabilité du pipeline | Rejet explicite ou traitement documenté ; impact des perturbations mesuré |
| T12 | Les performances techniques sont mesurées | Performance | Latence ou consommation excessive | Mesures de latence, débit, mémoire et taille du modèle documentées |

### 6.3. Correspondance avec les fichiers de test

Les identifiants T01 à T12 seront conservés dans les noms des tests automatisés.

| Exigence | Fichier ou support prévu | Exemple de test ou de contrôle |
|---|---|---|
| T01 | tests/test_preprocessing.py | test_T01_colonnes_requises_presentes |
| T02 | tests/test_preprocessing.py | test_T02_types_donnees_conformes |
| T03 | tests/test_preprocessing.py | test_T03_sortie_pretraitement_finie |
| T04 | tests/test_prediction.py | test_T04_probabilites_valides |
| T05 | tests/test_integration.py | test_T05_absence_fuite_donnees |
| T06 | tests/test_regression.py | test_T06_resultats_reproductibles |
| T07 | .github/workflows/ | Exécution de pytest et Ruff dans GitHub Actions |
| T08 | tests/test_preprocessing.py | test_T08_cible_binaire |
| T09 | tests/test_integration.py | test_T09_predictions_apres_rechargement |
| T10 | tests/test_regression.py | test_T10_detection_regression |
| T11 | tests/test_robustness.py | test_T11_entrees_invalides |
| T12 | Rapport de performance technique | Benchmark de latence, débit et mémoire |

Les noms ci-dessus représentent les tests et contrôles prévus. Leur existence et leur réussite devront être vérifiées lors des activités correspondantes.

### 6.4. Cas de test représentatifs

#### T01 — Vérification des colonnes requises

- **Précondition :** disposer d'un jeu de données conforme au schéma attendu.
- **Entrée valide :** données contenant toutes les colonnes obligatoires.
- **Entrée invalide :** données dont la colonne V12 a été supprimée.
- **Action :** exécuter `validate_data()`.
- **Résultat attendu :** acceptation des données valides et signalement explicite de la colonne manquante.
- **Preuve :** résultat du test pytest associé.

#### T03 — Vérification du prétraitement

- **Précondition :** disposer d'un pipeline de prétraitement configuré.
- **Entrée :** données numériques valides.
- **Action :** exécuter `preprocess_data()`.
- **Résultat attendu :** dimensions conformes et absence de NaN ou de valeurs infinies.
- **Preuve :** résultat du test unitaire et rapport de couverture.

#### T04 — Vérification des probabilités

- **Précondition :** disposer d'un modèle entraîné.
- **Entrée :** une ou plusieurs transactions valides.
- **Action :** exécuter `predict_fraud()`.
- **Résultat attendu :** une probabilité finie comprise entre 0 et 1 pour chaque transaction.
- **Preuve :** résultat du test fonctionnel ML.

#### T05 — Prévention des fuites de données

- **Précondition :** disposer des partitions d'entraînement, de validation et de test.
- **Action :** entraîner le pipeline, sélectionner le modèle et son seuil sur les partitions autorisées.
- **Résultat attendu :** aucune transformation apprise n'est ajustée sur le jeu de test ; le jeu de test n'intervient pas dans les choix de modèle, d'hyperparamètres ou de seuil.
- **Preuve :** test d'intégration et documentation du protocole expérimental.

#### T09 — Sauvegarde et rechargement

- **Précondition :** disposer d'un pipeline entraîné.
- **Action :** produire des prédictions, sauvegarder le pipeline, le recharger, puis prédire sur les mêmes observations.
- **Résultat attendu :** prédictions équivalentes dans la tolérance numérique définie.
- **Preuve :** résultat du test d'intégration.

#### T10 — Détection d'une régression

- **Précondition :** disposer d'une baseline documentée.
- **Action :** comparer les métriques d'une version candidate à celles de la baseline.
- **Résultat attendu :** toute dégradation dépassant les tolérances définies entraîne un échec du contrôle.
- **Preuve :** rapport de non-régression.

### 6.5. Suivi de la couverture des exigences

La couverture des exigences sera calculée comme suit :

**Couverture des exigences (%) = (Nombre d'exigences disposant de contrôles exécutés / Nombre total d'exigences) × 100**

Le projet comporte 12 exigences identifiées.

L'objectif est de disposer d'au moins un contrôle exécuté et d'une preuve associée pour chacune des 12 exigences.

La couverture ne doit pas être confondue avec le taux de réussite des tests : une exigence peut être couverte par un test exécuté mais non satisfait.

### 6.6. Gestion des résultats

Pour chaque exigence, le statut sera renseigné lors de l'exécution des tests :

- **Non exécuté** : aucun contrôle exécuté.
- **Réussi** : résultat conforme au critère attendu.
- **Échoué** : résultat non conforme.
- **Bloqué** : contrôle impossible à exécuter en raison d'une dépendance ou d'une anomalie.

Les résultats définitifs et les preuves seront consolidés dans le rapport final de qualification.

## 7. Stratégie et niveaux de test

## 7. Stratégie et niveaux de test

### 7.1. Approche générale

La stratégie de test d'AI-TestGuard repose sur une approche fondée sur les risques, associant des tests automatisés et des analyses expérimentales du modèle de détection de fraude.

Les tests seront exécutés progressivement afin de détecter les anomalies le plus tôt possible.

Les exigences T01 à T12 serviront de référence pour assurer la traçabilité entre les contrôles réalisés et les résultats obtenus.

### 7.2. Niveaux de test

| Niveau | Objectif | Exigences concernées |
|---|---|---|
| Tests unitaires | Vérifier individuellement les fonctions critiques | T01, T02, T03, T04, T08 |
| Tests d'intégration | Vérifier le fonctionnement du pipeline complet et sa sérialisation | T05, T09 |
| Tests de performance ML | Évaluer la qualité de la classification et analyser les erreurs | T05, T06, T10 |
| Tests de robustesse | Évaluer le comportement face aux entrées invalides et perturbées | T11 |
| Tests de non-régression | Comparer les résultats à une référence documentée | T06, T10 |
| Tests de performance technique | Mesurer la latence, le débit et la consommation de ressources | T12 |
| Tests d'intégration continue | Vérifier l'exécution automatisée des contrôles | T07 |

### 7.3. Stratégie d'évaluation du modèle

Deux algorithmes seront comparés :

- La régression logistique.
- L'arbre de décision.

Les données seront séparées en trois ensembles distincts :

- **Entraînement** : ajustement du modèle et des transformations.
- **Validation** : comparaison des configurations et sélection du seuil.
- **Test** : évaluation finale indépendante.

Les partitions aléatoires seront stratifiées lorsque cela est approprié afin de préserver la proportion de transactions frauduleuses.

Les transformations apprises seront ajustées uniquement sur les données d'entraînement.

Le jeu de test final ne sera pas utilisé pour choisir le modèle, régler ses hyperparamètres ou sélectionner son seuil.

### 7.4. Indicateurs d'évaluation

Les indicateurs suivants seront calculés :

- Précision (Precision).
- Rappel (Recall).
- F1-score.
- ROC-AUC.
- PR-AUC / Average Precision.
- Taux de faux positifs (FPR).
- Taux de faux négatifs (FNR).
- Matrice de confusion.

L'accuracy pourra être calculée à titre complémentaire, mais ne constituera pas le seul critère de qualité.

Une attention particulière sera accordée aux faux négatifs, qui correspondent aux fraudes non détectées, et aux faux positifs, qui correspondent aux transactions normales signalées à tort.

---

## 8. Environnement et outils

### 8.1. Environnement de développement

Le projet AI-TestGuard est développé dans un environnement Python isolé à l'aide d'un environnement virtuel `.venv`.

L'environnement de travail comprend :

- Windows et PowerShell.
- Visual Studio Code.
- Python et les bibliothèques définies dans `requirements.txt`.
- Git et GitHub pour le versionnement.
- GitHub Actions pour l'intégration continue.

Les versions exactes des dépendances seront conservées dans `requirements.txt`.

### 8.2. Outils de développement et de test

| Outil | Utilisation |
|---|---|
| pandas et NumPy | Manipulation et contrôle des données |
| scikit-learn | Prétraitement, entraînement et évaluation des modèles |
| pytest | Exécution des tests automatisés |
| pytest-cov | Mesure de la couverture de code |
| Ruff | Analyse statique et contrôle du style |
| joblib | Sauvegarde et rechargement du pipeline |
| matplotlib | Visualisation des métriques et courbes |
| psutil | Mesure de la consommation de ressources |
| Git | Gestion des versions |
| GitHub Actions | Automatisation des contrôles |

### 8.3. Jeu de données

Le projet utilise le jeu de données pédagogique *Credit Card Fraud Detection*, disponible sur Kaggle.

Il comprend des transactions bancaires anonymisées, avec :

- `Time` : variable temporelle ;
- `V1` à `V28` : composantes anonymisées issues d'une transformation PCA ;
- `Amount` : montant de la transaction ;
- `Class` : variable cible binaire.

La classe positive correspond aux transactions frauduleuses.

Le jeu de données est fortement déséquilibré. Les métriques seront donc interprétées en tenant compte de la rareté des fraudes.

Le fichier CSV complet ne sera pas ajouté au dépôt GitHub. Les instructions permettant de récupérer les données seront documentées dans `data/README.md`.

### 8.4. Reproductibilité

Les expériences devront documenter :

- Les versions logicielles.
- Les paramètres des modèles.
- Les graines aléatoires.
- Les proportions des partitions.
- Les métriques et seuils utilisés.
- Les tolérances des tests de non-régression.

---

## 9. Critères d'acceptation et de rejet

### 9.1. Principe général

L'acceptation technique du pipeline dépendra de la conformité aux exigences T01 à T12 et de l'analyse des résultats obtenus.

Les critères seront définis avant l'évaluation finale sur le jeu de test.

La réussite des tests techniques ne constituera pas, à elle seule, une autorisation de déploiement en production bancaire.

### 9.2. Critères d'acceptation fonctionnelle

Le pipeline pourra être considéré comme fonctionnellement conforme si :

1. Les données invalides sont détectées ou traitées selon les règles documentées.
2. Les données prétraitées respectent les dimensions et les types attendus.
3. Les sorties numériques sont finies et les probabilités appartiennent à l'intervalle [0, 1].
4. Les prédictions restent cohérentes après sauvegarde et rechargement du pipeline.
5. Les tests unitaires et d'intégration critiques réussissent.

### 9.3. Critères d'acceptation méthodologique

L'évaluation sera considérée comme méthodologiquement conforme si :

1. Les données d'entraînement, de validation et de test sont distinctes.
2. Les transformations apprises sont ajustées exclusivement sur les données d'entraînement.
3. Le jeu de test final n'intervient pas dans le choix du modèle, de ses hyperparamètres ou de son seuil.
4. Les expériences sont reproductibles dans les tolérances définies.
5. Les métriques obligatoires sont calculées et interprétées.
6. Les faux positifs et les faux négatifs font l'objet d'une analyse spécifique.

### 9.4. Critères d'acceptation de la performance ML

La performance prédictive sera évaluée à partir de la précision, du rappel, du F1-score, de la ROC-AUC, de la PR-AUC et des taux de faux positifs et de faux négatifs.

Le modèle candidat sera comparé à une référence documentée.

Les seuils minimaux de performance, les tolérances de non-régression et les contraintes opérationnelles devront être fixés sur la base des objectifs de qualification, des résultats de validation et des coûts métier considérés, avant l'évaluation finale.

Aucun seuil chiffré de rappel, de précision ou de F1-score n'est présumé atteint à ce stade.

### 9.5. Critères d'acceptation technique

Les contrôles techniques devront démontrer que :

- Les tests automatisés peuvent être exécutés localement.
- GitHub Actions exécute les contrôles prévus et signale correctement les échecs.
- Les rapports de couverture et d'analyse statique sont disponibles.
- Les mesures de latence, débit, taille du modèle et mémoire sont documentées.
- Les anomalies critiques identifiées sont corrigées ou justifient explicitement un rejet.

### 9.6. Critères de rejet ou de qualification conditionnelle

La qualification technique sera rejetée ou suspendue notamment si :

- Une fuite de données compromet l'évaluation.
- Les fonctions critiques produisent des résultats invalides.
- Les tests critiques échouent sans justification ni correction.
- Les résultats ne sont pas reproductibles dans les tolérances prévues.
- Une dégradation importante du modèle n'est pas maîtrisée.
- Les preuves nécessaires à la qualification sont absentes.

Une qualification conditionnelle pourra être proposée si les anomalies restantes sont non critiques, documentées et accompagnées de mesures correctives et de limites d'utilisation explicites.

### 9.7. Décision finale

La conclusion distinguera :

- **Conforme techniquement** : les contrôles et critères techniques définis sont satisfaits.
- **Conforme sous conditions** : certaines réserves documentées nécessitent des actions complémentaires.
- **Non conforme** : une ou plusieurs défaillances critiques empêchent la qualification.

L'aptitude à poursuivre vers un pilote sera appréciée séparément.

Une mise en production réelle nécessiterait également des validations métier, opérationnelles, sécuritaires et réglementaires.

---

## 10. Gestion des anomalies

### 10.1. Objectif

Toute anomalie détectée pendant les tests devra être enregistrée, analysée et suivie jusqu'à sa résolution ou son acceptation justifiée.

### 10.2. Informations à enregistrer

Chaque anomalie sera décrite à l'aide des informations suivantes :

- Identifiant de l'anomalie.
- Exigence concernée (T01 à T12).
- Description du problème.
- Étapes de reproduction.
- Résultat attendu.
- Résultat observé.
- Gravité.
- Statut.
- Action corrective et résultat du nouveau test.

### 10.3. Classification de la gravité

| Gravité | Définition |
|---|---|
| Critique | Compromet la validité des résultats, la sécurité ou le fonctionnement essentiel |
| Majeure | Affecte une fonctionnalité importante ou une mesure de qualification |
| Mineure | Affecte un comportement secondaire sans compromettre les résultats principaux |

### 10.4. Statuts des anomalies

Les statuts utilisés seront :

- Ouverte.
- En cours de correction.
- Corrigée.
- Vérifiée.
- Acceptée avec réserve, si justifié.

Les anomalies non résolues seront présentées dans le rapport final avec leurs risques résiduels.

---

## 11. Preuves et livrables

### 11.1. Preuves à conserver

Les preuves d'exécution comprendront notamment :

- Résultats des tests pytest.
- Rapport de couverture.
- Résultats de l'analyse Ruff.
- Métriques ML et matrices de confusion.
- Courbes ROC et précision-rappel.
- Résultats des tests de robustesse.
- Baseline et résultats de non-régression.
- Mesures de latence et de consommation de ressources.
- Journaux et captures des exécutions GitHub Actions.
- Registre des anomalies.

### 11.2. Livrables attendus

Les livrables du projet seront :

1. Un dépôt GitHub contenant le code, les tests, les dépendances et la documentation.
2. Un plan de test avec exigences, risques, cas de test et critères d'acceptation.
3. Une suite de tests automatisés couvrant les fonctions critiques et le pipeline.
4. Un rapport d'évaluation ML avec métriques, graphiques et analyse des erreurs.
5. Un rapport de robustesse et de non-régression.
6. Un workflow GitHub Actions opérationnel, accompagné des preuves d'exécution.
7. Un rapport final de qualification avec une conclusion argumentée.

Les preuves devront être identifiables et cohérentes avec les exigences T01 à T12.

---

## 12. Limites et hypothèses

### 12.1. Limites des données

Le jeu de données est anonymisé et ne contient pas les attributs démographiques nécessaires à une analyse complète de l'équité du modèle entre groupes de population.

Les composantes V1 à V28 ne seront pas interprétées comme des variables métier directement compréhensibles.

### 12.2. Limites de généralisation

Les performances obtenues sur ce jeu de données pédagogique ne démontrent pas que le modèle serait aussi performant sur des transactions bancaires actuelles.

Une évaluation complémentaire sur des données récentes et représentatives serait nécessaire avant toute utilisation réelle.

### 12.3. Limites opérationnelles

Le projet ne comprend pas :

- Une intégration à un système bancaire réel.
- Un déploiement en production.
- Une validation réglementaire ou métier complète.
- Une démonstration exhaustive de la sécurité d'un environnement bancaire réel.

### 12.4. Hypothèses de travail

Les hypothèses suivantes sont retenues :

- Le jeu de données utilisé est conforme à sa description.
- Les expériences sont exécutées dans un environnement logiciel documenté.
- Les partitions et les paramètres sont conservés pour assurer la reproductibilité.
- Les critères de qualification sont définis avant l'utilisation du jeu de test final.
- Les résultats et anomalies sont enregistrés de manière traçable.

### 12.5. Risques résiduels

Les principaux risques résiduels susceptibles de subsister après les tests concernent :

- L'évolution des distributions des transactions.
- La dégradation du modèle dans le temps.
- Les biais impossibles à mesurer avec les attributs disponibles.
- Les erreurs de classification ayant des conséquences métier.
- Les limites de représentativité du jeu de données.

Ces risques seront examinés dans le rapport final et dans la stratégie de surveillance prévue à l'activité 10.