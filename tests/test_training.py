
import numpy as np
import pandas as pd
import pytest

from app.training import train_model


@pytest.fixture
def donnees_entrainement():
    rng = np.random.default_rng(42)

    colonnes = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    X = pd.DataFrame(
        rng.normal(size=(20, 30)),
        columns=colonnes,
    )

    # Le montant ne peut pas être négatif.
    X["Amount"] = X["Amount"].abs() * 100

    # Cible binaire : 10 transactions normales,
    # 10 transactions frauduleuses.
    y = pd.Series([0] * 10 + [1] * 10)

    return X, y


def test_entrainement_modele(donnees_entrainement):
    """Vérifie que le modèle peut être entraîné."""
    X, y = donnees_entrainement

    model = train_model(X, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert model.n_features_in_ == 30


def test_probabilites_apres_entrainement():
    """Vérifie la validité des probabilités produites."""

    X = pd.DataFrame(
        {
            "V1": [-2.0, -1.0, 1.0, 2.0],
            "Amount": [10.0, 20.0, 100.0, 200.0],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    model = train_model(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
