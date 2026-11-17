
import numpy as np
import pandas as pd
import pytest

from app.training import split_data, train_model


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

def test_T05_proportions_partitionnement():
    """Vérifier la répartition 60/20/20."""
    X = pd.DataFrame({"V1": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20

    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20

def test_T05_absence_chevauchement():
    """Aucune observation ne doit appartenir à deux ensembles."""
    X = pd.DataFrame({"V1": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    X_train, X_val, X_test, _, _, _ = split_data(X, y)

    train_ids = set(X_train.index)
    val_ids = set(X_val.index)
    test_ids = set(X_test.index)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert train_ids | val_ids | test_ids == set(X.index)


def test_T05_stratification():
    """Vérifier la conservation de la proportion des classes."""
    X = pd.DataFrame({"V1": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    _, _, _, y_train, y_val, y_test = split_data(X, y)

    taux_initial = y.mean()

    assert abs(y_train.mean() - taux_initial) < 0.05
    assert abs(y_val.mean() - taux_initial) < 0.05
    assert abs(y_test.mean() - taux_initial) < 0.05


def test_T06_partitionnement_reproductible():
    """Une même graine doit produire les mêmes partitions."""

    X = pd.DataFrame({"V1": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    resultat_1 = split_data(X, y, random_state=42)
    resultat_2 = split_data(X, y, random_state=42)

    for partie_1, partie_2 in zip(resultat_1, resultat_2):
        if isinstance(partie_1, pd.DataFrame):
            pd.testing.assert_frame_equal(partie_1, partie_2)
        else:
            pd.testing.assert_series_equal(partie_1, partie_2)


