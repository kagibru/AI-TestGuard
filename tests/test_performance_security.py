
# FICHIER : tests/test_performance_security.py

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import preprocess_data, validate_data
from app.training import train_model


@pytest.fixture
def trained_model_T12():
    """Créer un modèle minimal pour les tests de sécurité."""

    X = pd.DataFrame({
        "V1": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "Amount": [10, 20, 30, 100, 200, 300],
    })

    y = pd.Series([0, 0, 0, 1, 1, 1])

    return train_model(X, y), X


def test_T12_probabilites_finies(trained_model_T12):
    """Les probabilités doivent être finies et bornées."""

    model, X = trained_model_T12

    probabilities = model.predict_proba(X)

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_T12_rejet_schema_invalide():
    """Les données incomplètes doivent être rejetées."""

    df = pd.DataFrame({
        "Amount": [100.0],
        "Class": [0],
    })

    with pytest.raises(ValueError):
        validate_data(df)


def test_T12_rejet_entree_non_numerique():
    """Le contrat de données rejette les valeurs textuelles."""

    df = pd.DataFrame({
        "Time": [0.0],
        **{f"V{i}": [0.1] for i in range(1, 29)},
        "Amount": ["DONNEE_CONFIDENTIELLE_TEST"],
        "Class": [0],
    })

    with pytest.raises((TypeError, ValueError)) as error:
        preprocess_data(df)

    # Le message ne doit pas reproduire la valeur d'entrée.
    assert "DONNEE_CONFIDENTIELLE_TEST" not in str(error.value)
