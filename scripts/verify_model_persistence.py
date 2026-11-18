from pathlib import Path

import numpy as np
import pandas as pd

from app.persistence import load_model, save_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data
from app.training import split_data, train_model

MODEL_PATH = Path("models/fraud_logistic_regression.joblib")
THRESHOLD = 0.90


def main():
    # 1. Charger les données réelles.
    print("=== Chargement des données Kaggle ===")

    df = pd.read_csv("data/creditcard.csv")
    X, y = preprocess_data(df)

    # 2. Partitionner les données.
    X_train, X_val, X_test, y_train, _y_val, _y_test = (
        split_data(X, y, random_state=42)
    )

    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test réservé : {len(X_test):,}")

    # 3. Entraîner uniquement sur Train.
    print("\n=== Entraînement ===")

    model = train_model(X_train, y_train)

    # 4. Prédire AVANT sauvegarde.
    probabilities_before, predictions_before = predict_fraud(
        model, X_val, threshold=THRESHOLD
    )

    # 5. Sauvegarder le pipeline.
    print("\n=== Sauvegarde ===")

    save_model(model, MODEL_PATH)

    print(f"Modèle sauvegardé : {MODEL_PATH}")

    # 6. Recharger le pipeline.
    print("\n=== Rechargement ===")

    loaded_model = load_model(MODEL_PATH)

    # 7. Prédire APRÈS rechargement.
    probabilities_after, predictions_after = predict_fraud(
        loaded_model, X_val, threshold=THRESHOLD
    )

    # 8. Vérifier les probabilités.
    np.testing.assert_allclose(
        probabilities_before,
        probabilities_after,
        rtol=0,
        atol=1e-12,
    )

    # 9. Vérifier les classes prédites.
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )

    # 10. Vérifier les paramètres du scaler.
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_,
        loaded_model.named_steps["scaler"].mean_,
        rtol=0,
        atol=0,
    )

    print("\n=== Résultat T09 ===")
    print("Probabilités identiques : OUI")
    print("Prédictions identiques  : OUI")
    print("Paramètres scaler identiques : OUI")
    print("T09 : VALIDÉE sur Kaggle (Validation)")
    print("Le jeu de test final n'a pas été utilisé.")


if __name__ == "__main__":
    main()
