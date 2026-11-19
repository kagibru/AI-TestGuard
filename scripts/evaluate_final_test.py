
"""
Évaluation finale hors échantillon du modèle AI-TestGuard.

Le modèle et le seuil ont été sélectionnés sur Validation.
Cette évaluation utilise exclusivement Test, sans réentraînement.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocessing import preprocess_data
from app.training import split_data


DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/fraud_logistic_regression.joblib")
REPORT_PATH = Path("reports/final_test_evaluation.json")

RANDOM_STATE = 42
THRESHOLD = 0.90


def main():
    # 1. Vérifier la disponibilité des fichiers.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Jeu de données introuvable : {DATA_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle sauvegardé introuvable : {MODEL_PATH}"
        )

    # 2. Reconstituer le découpage initial.
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y, random_state=RANDOM_STATE)
    )

    # 3. Vérifier l'indépendance des partitions.
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    print("=== ÉVALUATION FINALE HORS ÉCHANTILLON ===")
    print(f"Entraînement : {len(X_train)}")
    print(f"Validation   : {len(X_val)}")
    print(f"Test         : {len(X_test)}")
    print(f"Fraudes Test : {int(y_test.sum())}")

    # 4. Charger le modèle existant : aucun réentraînement.
    # Charger uniquement un fichier joblib local de confiance.
    model = joblib.load(MODEL_PATH)

    # 5. Prédire avec le seuil fixé sur Validation.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # 6. Calculer les métriques.
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    report = {
        "model": "LogisticRegression",
        "evaluation_set": "test",
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "model_path": str(MODEL_PATH),
        "model_retrained": False,
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "test_class_distribution": {
            "normal": int((y_test == 0).sum()),
            "fraud": int((y_test == 1).sum()),
        },
        "metrics_test": metrics,
    }

    # 7. Enregistrer la preuve.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"\nSeuil fixé : {THRESHOLD}")
    print("\n=== MÉTRIQUES FINALES ===")

    for name, value in metrics.items():
        if name in {"tn", "fp", "fn", "tp"}:
            print(f"{name:12}: {value}")
        else:
            print(f"{name:12}: {value:.4f}")

    print(f"\nRapport enregistré : {REPORT_PATH}")


if __name__ == "__main__":
    main()
