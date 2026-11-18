"""
Comparaison de deux modèles de détection de fraude :

1. Régression logistique
2. Arbre de décision

Les deux modèles utilisent :
- Le même jeu d'entraînement
- Le même jeu de validation
- Les mêmes seuils de classification

Le jeu de test final reste inutilisé.
"""

import json
from pathlib import Path

import pandas as pd

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data
from app.training import (
    split_data,
    train_decision_tree,
    train_model,
)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

DATA_PATH = Path("data/creditcard.csv")

REPORT_DIR = Path("reports")

RANDOM_STATE = 42

THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]


# -------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------

def main():

    # 1. Charger les données.
    print("=== Chargement des données ===")

    df = pd.read_csv(DATA_PATH)

    print(f"Transactions chargées : {len(df):,}")

    # 2. Prétraiter les données.
    X, y = preprocess_data(df)

    # 3. Partitionner les données.
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(
            X,
            y,
            random_state=RANDOM_STATE,
        )
    )

    print("\n=== Répartition ===")

    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test       : {len(X_test):,}")

    print(f"\nFraudes Train      : {y_train.mean():.4%}")
    print(f"Fraudes Validation : {y_val.mean():.4%}")
    print(f"Fraudes Test       : {y_test.mean():.4%}")

    # 4. Entraîner les modèles sur Train uniquement.
    print("\n=== Entraînement des modèles ===")

    print("Entraînement : régression logistique...")

    logistic_model = train_model(
        X_train,
        y_train,
    )

    print("Entraînement : arbre de décision...")

    tree_model = train_decision_tree(
        X_train,
        y_train,
    )

    # 5. Regrouper les modèles entraînés.
    models = {
        "LogisticRegression": logistic_model,
        "DecisionTree": tree_model,
    }

    # 6. Évaluer les modèles.
    results = []

    print("\n=== Évaluation sur Validation ===")

    for model_name, model in models.items():

        print(f"\nModèle : {model_name}")

        for threshold in THRESHOLDS:

            # Produire les probabilités et décisions.
            probabilities, predictions = predict_fraud(
                model,
                X_val,
                threshold=threshold,
            )

            # Calculer les métriques.
            metrics = evaluate_model(
                y_val,
                predictions,
                probabilities,
            )

            # Enregistrer une ligne de résultats.
            results.append({
                "model": model_name,
                "threshold": threshold,
                **metrics,
            })

    # 7. Construire le tableau comparatif.
    results_df = pd.DataFrame(results)

    selected_columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
    ]

    print("\n=== Tableau comparatif ===")

    print(
        results_df[selected_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # 8. Enregistrer les résultats.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = REPORT_DIR / "model_comparison.csv"
    json_path = REPORT_DIR / "model_comparison.json"

    results_df.to_csv(
        csv_path,
        index=False,
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n=== Fichiers générés ===")
    print(csv_path)
    print(json_path)

    print(
        "\nComparaison terminée. "
        "Le jeu de test final n'a pas été utilisé."
    )


if __name__ == "__main__":
    main()
