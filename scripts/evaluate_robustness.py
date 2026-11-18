import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data
from app.training import split_data, train_model

THRESHOLD = 0.90
RANDOM_STATE = 42


def evaluate_scenario(model, X, y, name):
    """Évaluer un scénario et retourner ses métriques."""

    probabilities, predictions = predict_fraud(
        model, X, threshold=THRESHOLD
    )

    metrics = evaluate_model(y, predictions, probabilities)

    return {
        "scenario": name,
        **metrics,
    }


def main():
    print("=== Évaluation de robustesse ===")

    # 1. Charger et préparer les données.
    df = pd.read_csv("data/creditcard.csv")
    X, y = preprocess_data(df)

    # 2. Partitionner les données.
    X_train, X_val, _X_test, y_train, y_val, _y_test = (
        split_data(X, y, random_state=RANDOM_STATE)
    )

    # 3. Entraîner sur Train uniquement.
    model = train_model(X_train, y_train)

    results = []

    # SCÉNARIO A : données originales.
    results.append(
        evaluate_scenario(
            model, X_val, y_val, "reference"
        )
    )

    # SCÉNARIO B : légère perturbation des variables V1-V28.
    rng = np.random.default_rng(RANDOM_STATE)

    X_noise = X_val.copy()

    feature_columns = [f"V{i}" for i in range(1, 29)]

    train_std = X_train[feature_columns].std().to_numpy()

    noise = rng.normal(
        loc=0.0,
        scale=0.01 * train_std,
        size=(len(X_noise), len(feature_columns)),
    )

    X_noise.loc[:, feature_columns] = (
        X_noise[feature_columns].to_numpy() + noise
    )

    results.append(
        evaluate_scenario(
            model, X_noise, y_val, "bruit_1pct_ecart_type"
        )
    )

    # SCÉNARIO C : augmentation contrôlée des montants.
    X_amount = X_val.copy()

    X_amount["Amount"] = X_amount["Amount"] * 1.20

    results.append(
        evaluate_scenario(
            model, X_amount, y_val, "montant_plus_20pct"
        )
    )

    # 4. Comparer les résultats.
    results_df = pd.DataFrame(results)

    reference = results_df.iloc[0]

    results_df["delta_recall"] = (
        results_df["recall"] - reference["recall"]
    )

    results_df["delta_precision"] = (
        results_df["precision"] - reference["precision"]
    )

    results_df["delta_f1"] = (
        results_df["f1"] - reference["f1"]
    )

    print("\n=== Résultats ===")

    columns = [
        "scenario",
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "delta_recall",
        "delta_precision",
        "delta_f1",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # 5. Sauvegarder les preuves.
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_dir / "robustness_results.csv",
        index=False,
    )

    with (
        output_dir / "robustness_results.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nRésultats enregistrés dans reports/")
    print("Le jeu de test final n'a pas été utilisé.")


if __name__ == "__main__":
    main()
