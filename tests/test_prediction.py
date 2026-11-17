
import numpy as np
import pandas as pd
import pytest

from app.prediction import predict_fraud


class FakeModel:
    """Modèle simulé renvoyant des probabilités connues."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probabilities = np.array([0.03, 0.78, 0.42])
        probabilities = probabilities[: len(X)]

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {"Amount": [100.0, 250.0, 75.0]}
    )


def test_T04_probabilites_valides(transactions: pd.DataFrame):
    probabilities, predictions = predict_fraud(FakeModel(), transactions)

    assert len(probabilities) == 3
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert predictions.tolist() == [0, 1, 0]


def test_T04_seuil_personnalise(transactions: pd.DataFrame):
    _, predictions = predict_fraud(
        FakeModel(), transactions, threshold=0.4
    )

    assert predictions.tolist() == [0, 1, 1]


def test_T04_seuil_invalide(transactions: pd.DataFrame):
    with pytest.raises(ValueError):
        predict_fraud(
            FakeModel(), transactions, threshold=1.5
        )


def test_T04_entree_vide():
    with pytest.raises(ValueError):
        predict_fraud(FakeModel(), pd.DataFrame())


def test_T04_entree_non_dataframe():
    with pytest.raises(TypeError):
        predict_fraud(FakeModel(), [100, 250, 75])
