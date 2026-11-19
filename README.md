
# AI-TestGuard

## 1. Présentation du projet

**AI-TestGuard** est un projet de qualification logicielle d'un modèle de Machine Learning destiné à la détection de transactions bancaires frauduleuses.

Réalisé dans le cadre du **Master 2 Data Science et Intelligence Artificielle (MDSIA)**, il vise à évaluer la qualité, la fiabilité, la robustesse, la reproductibilité et les performances techniques d'un système de prédiction.

L'objectif n'est pas uniquement d'entraîner un modèle performant, mais de mettre en œuvre une démarche structurée de tests logiciels permettant de documenter son aptitude à poursuivre vers une expérimentation pilote.

## 2. Objectifs

- Vérifier la qualité et la cohérence des données.
- Développer des tests unitaires et d'intégration.
- Évaluer les performances prédictives des modèles.
- Tester leur robustesse face à des perturbations.
- Vérifier la reproductibilité et la non-régression.
- Identifier les biais potentiels et les limites méthodologiques.
- Mesurer la latence et la consommation de ressources.
- Vérifier les contrôles de validation des entrées.
- Automatiser les tests avec GitHub Actions.
- Documenter les résultats et les conditions de surveillance.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.13 | Développement |
| pandas et NumPy | Manipulation des données |
| scikit-learn | Prétraitement, entraînement et évaluation |
| joblib | Persistance du modèle |
| pytest | Tests automatisés |
| pytest-cov | Couverture des tests |
| Ruff | Analyse statique du code |
| psutil | Mesure des ressources |
| Git et GitHub | Gestion des versions |
| GitHub Actions | Intégration continue |

## 4. Jeu de données

**Source :** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

Le jeu de données comprend **284 807 transactions**, dont **492 frauduleuses**.

La variable cible `Class` distingue :

- `0` : transaction normale ;
- `1` : transaction frauduleuse.

Les données sont fortement déséquilibrées. Les variables `V1` à `V28` sont anonymisées, ce qui limite leur interprétation métier.

Le fichier `creditcard.csv` n'est pas inclus dans le dépôt GitHub.

## 5. Architecture du projet

```text
AI-TestGuard/
├── app/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── persistence.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_prediction.py
│   ├── test_evaluation.py
│   ├── test_integration.py
│   ├── test_robustness.py
│   ├── test_regression.py
│   ├── test_persistence.py
│   └── test_performance_security.py
├── scripts/
│   ├── evaluate_validation.py
│   ├── compare_models.py
│   ├── verify_model_persistence.py
│   ├── evaluate_robustness.py
│   └── benchmark_model.py
├── reports/
├── models/
├── data/
├── .github/
│   └── workflows/
│       └── tests.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

## 6. Installation

### 6.1. Cloner le dépôt

```powershell
git clone https://github.com/kagibru/AI-TestGuard.git
cd AI-TestGuard
```

### 6.2. Créer un environnement virtuel

```powershell
python -m venv .venv
```

### 6.3. Activer l'environnement sous Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6.4. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 6.5. Préparer les données

Télécharger le fichier `creditcard.csv` depuis Kaggle et le placer dans le répertoire :

```text
data/creditcard.csv
```

Le fichier de données n'est pas distribué avec le dépôt.

## 7. Exécution des tests

### 7.1. Exécuter tous les tests localement

```powershell
python -m pytest -v
```

Les tests nécessitant le jeu de données Kaggle supposent que le fichier `data/creditcard.csv` est disponible.

### 7.2. Exécuter les tests sélectionnés pour la CI

```powershell
python -m pytest -m "not integration" -v
```

### 7.3. Mesurer la couverture

```powershell
python -m pytest -m "not integration" --cov=app --cov-branch --cov-report=term-missing
```

### 7.4. Vérifier la qualité du code

```powershell
ruff check app tests scripts
```

## 8. Modèles et résultats prédictifs

Deux modèles de classification ont été comparés :

1. Une régression logistique pondérée.
2. Un arbre de décision pondéré.

Le modèle retenu est la **régression logistique**, avec un seuil de décision de **0,90**, sélectionné sur le jeu de validation.

### Résultats sur le jeu de validation

| Indicateur | Valeur |
|---|---:|
| Précision | 0,2516 |
| Rappel | 0,8265 |
| F1-score | 0,3857 |
| ROC-AUC | 0,9806 |
| PR-AUC | 0,6857 |
| Faux positifs | 241 |
| Faux négatifs | 17 |

Ces résultats ne constituent pas une validation définitive sur le jeu de test indépendant.

## 9. Robustesse et reproductibilité

Les scénarios de robustesse étudiés comprennent :

- Une perturbation aléatoire de 1 % de l'écart-type.
- Une augmentation de 20 % du montant des transactions.

Les tests permettent d'observer les variations des métriques prédictives et d'identifier les sensibilités du modèle.

La reproductibilité repose notamment sur :

- Une graine aléatoire fixe (`random_state=42`).
- Un découpage stratifié des données.
- Des tests de non-régression.
- La vérification de la persistance du modèle avec `joblib`.

Les résultats détaillés sont conservés dans le répertoire `reports/`.

## 10. Performance technique — T12

Un benchmark a été réalisé sur le jeu de validation, avec **100 itérations portant chacune sur 100 transactions**, après 10 exécutions de préchauffage.

| Indicateur | Résultat |
|---|---:|
| Taille du lot | 100 transactions |
| Nombre d'itérations | 100 |
| Préchauffages | 10 |
| Latence moyenne par lot | 0,567 ms |
| Latence P95 par lot | 0,614 ms |
| Débit mesuré | 176 244 transactions/s |
| Taille du modèle | 2,66 Ko |
| Variation nette de mémoire | 0 Mo |

Ces mesures correspondent à une inférence locale en mémoire, sans infrastructure API ni concurrence entre utilisateurs.

La variation nette de mémoire de 0 Mo ne signifie pas une absence de consommation mémoire.

**Preuves :**

- `scripts/benchmark_model.py`
- `reports/benchmark_model.json`
- `tests/test_performance_security.py`

## 11. Intégration continue

Le projet utilise **GitHub Actions** pour automatiser :

- L'installation des dépendances.
- L'analyse statique du code avec Ruff.
- L'exécution des tests sélectionnés avec pytest.
- La vérification d'un seuil minimal de couverture de 85 %.
- La génération des rapports de tests.

Le workflow a été exécuté avec succès le **19 novembre 2026 (run #4)**.

Les tests d'intégration dépendant du fichier Kaggle sont exclus de la CI automatique et doivent être exécutés localement avec les données disponibles.

Le fichier de configuration du workflow est :

```text
.github/workflows/tests.yml
```

## 12. Limites et risques résiduels

Plusieurs limites doivent être prises en compte :

- Le déséquilibre important entre les classes.
- Le coût opérationnel potentiel des faux positifs.
- Le risque associé aux fraudes non détectées.
- Les variables anonymisées limitant l'analyse des biais.
- L'absence de validation en environnement de production.
- L'absence de mesures de performance sous charge concurrente.
- La nécessité de surveiller les dérives des données et des performances.

La réussite des tests automatisés ne constitue pas, à elle seule, une autorisation de déploiement en production.

## 13. Documentation et preuves

Les principaux éléments de preuve sont regroupés dans le répertoire `reports/`.

Ils comprennent notamment :

- `plan_de_test.md` : plan de test et matrice de traçabilité.
- `validation_logistic_regression.json` : résultats de validation.
- `model_comparison.json` : comparaison des modèles.
- `robustness_results.json` : résultats de robustesse.
- `baseline_model.json` : référence de non-régression.
- `analyse_biais_limites.md` : analyse des biais et limites.
- `benchmark_model.json` : mesures de performance technique.

Le rapport final de qualification complète ces éléments de preuve.

## 14. Conclusion

AI-TestGuard démontre l'application d'une démarche structurée de qualification logicielle à un système de Machine Learning.

Les résultats prédictifs, les tests automatisés et les mesures techniques fournissent des éléments encourageants pour envisager une expérimentation pilote encadrée.

Cette perspective reste conditionnée à la validation des risques métier, des exigences de sécurité, des conditions opérationnelles et des critères d'acceptation du pilote.

---

**Projet académique — Master 2 Data Science et Intelligence Artificielle (MDSIA)**  
**Date : 19 novembre 2026**