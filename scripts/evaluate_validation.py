
"""
Évaluation de la régression logistique sur le jeu de validation.

Objectifs :
- Charger et préparer les données Kaggle.
- Vérifier la répartition des classes.
- Entraîner le modèle uniquement sur Train.
- Évaluer le modèle sur Validation.
- Comparer plusieurs seuils de classification.
- Enregistrer les résultats pour le rapport de qualification.

Le jeu de test final reste inutilisé.
"""

import json
from pathlib import Path

import pandas as pd

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data
from app.training import split_data, train_model

DATA_PATH = Path("data/creditcard.csv")

REPORT_DIR = Path("reports")

BASELINE_THRESHOLD = 0.50

THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]


def display_distribution(name, y):
    """Afficher la distribution des classes d'un ensemble."""

    normal_count = int((y == 0).sum())
    fraud_count = int((y == 1).sum())
    fraud_rate = float(y.mean())

    print(
        f"{name:12} | "
        f"Total : {len(y):7,} | "
        f"Normales : {normal_count:7,} | "
        f"Fraudes : {fraud_count:4,} | "
        f"Taux fraude : {fraud_rate:.4%}"
    )


def display_metrics(metrics):
    """Afficher les métriques calculées."""

    print("\n=== Performances sur Validation ===")

    for name, value in metrics.items():
        if name in {"tn", "fp", "fn", "tp"}:
            print(f"{name:12}: {value}")
        else:
            print(f"{name:12}: {value:.4f}")


def compare_thresholds(model, X_val, y_val):
    """Comparer les seuils sur le jeu de validation."""

    print("\n=== Comparaison des seuils ===")

    results = []

    for threshold in THRESHOLDS:

        probabilities, predictions = predict_fraud(
            model,
            X_val,
            threshold=threshold,
        )

        metrics = evaluate_model(
            y_val,
            predictions,
            probabilities,
        )

        results.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return results_df


def main():

    # =====================================================
    # 1. Chargement des données
    # =====================================================

    print("=== Chargement des données Kaggle ===")

    df = pd.read_csv(DATA_PATH)

    print(f"Transactions chargées : {len(df):,}")

    # =====================================================
    # 2. Validation et prétraitement
    # =====================================================

    X, y = preprocess_data(df)

    # =====================================================
    # 3. Partitionnement stratifié
    # =====================================================

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(
            X,
            y,
            random_state=42,
        )
    )

    print("\n=== Répartition des données ===")

    display_distribution("Global", y)
    display_distribution("Train", y_train)
    display_distribution("Validation", y_val)
    display_distribution("Test", y_test)

    # Vérifier que les partitions ne se chevauchent pas.
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    # Vérifier que les proportions restent comparables.
    global_fraud_rate = float(y.mean())
    tolerance = 0.0001

    for target in (y_train, y_val, y_test):
        assert abs(float(target.mean()) - global_fraud_rate) <= tolerance

    # =====================================================
    # 4. Entraînement du modèle
    # =====================================================

    print("\n=== Entraînement ===")
    print("Modèle : régression logistique")
    print("Ensemble utilisé : Train uniquement")

    model = train_model(X_train, y_train)

    # =====================================================
    # 5. Évaluation initiale : seuil 0.50
    # =====================================================

    probabilities, predictions = predict_fraud(
        model,
        X_val,
        threshold=BASELINE_THRESHOLD,
    )

    metrics = evaluate_model(
        y_val,
        predictions,
        probabilities,
    )

    display_metrics(metrics)

    # =====================================================
    # 6. Comparaison des seuils
    # =====================================================

    thresholds_df = compare_thresholds(
        model,
        X_val,
        y_val,
    )

    # =====================================================
    # 7. Enregistrement des résultats
    # =====================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": "LogisticRegression",
        "dataset": DATA_PATH.name,
        "random_state": 42,
        "initial_threshold": BASELINE_THRESHOLD,
        "class_weight": "balanced",
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "fraud_rates": {
            "train": float(y_train.mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "metrics_validation": metrics,
        "threshold_comparison": thresholds_df.to_dict(
            orient="records"
        ),
    }

    json_path = (
        REPORT_DIR / "validation_logistic_regression.json"
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    csv_path = (
        REPORT_DIR / "validation_threshold_comparison.csv"
    )

    thresholds_df.to_csv(
        csv_path,
        index=False,
    )

    print("\n=== Fichiers générés ===")
    print(json_path)
    print(csv_path)

    print(
        "\nÉvaluation terminée. "
        "Le jeu de test final n'a pas été utilisé."
    )


if __name__ == "__main__":
    main()
