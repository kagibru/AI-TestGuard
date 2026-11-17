
import numpy as np
import pandas as pd


def predict_fraud(model, X, threshold=0.5):
    """
    Produit les probabilités et décisions de fraude.

    Args:
        model : pipeline scikit-learn entraîné.
        X : DataFrame des variables explicatives.
        threshold : seuil de décision entre 0 et 1.

    Returns:
        probabilities : probabilités de fraude.
        predictions : classes prédites (0 ou 1).
    """

    if not 0 < threshold < 1:
        raise ValueError("Le seuil doit être strictement compris entre 0 et 1.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame pandas.")

    if X.empty:
        raise ValueError("Le jeu de données est vide.")

    probabilities = model.predict_proba(X)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilités non finies.")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilités hors de l'intervalle [0, 1].")

    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions
