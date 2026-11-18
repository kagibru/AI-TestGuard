import numpy as np
import pandas as pd
import pytest

from app.preprocessing import validate_data

# FICHIER : tests/test_robustness.py
# AJOUTER CES TESTS AUX TESTS EXISTANTS

@pytest.fixture
def valid_transactions_T11():
    """Jeu minimal conforme au schéma Kaggle."""

    data = {
        "Time": [0.0, 1.0, 2.0],
        **{
            f"V{i}": [0.1, 0.2, 0.3]
            for i in range(1, 29)
        },
        "Amount": [10.0, 100.0, 200.0],
        "Class": [0, 1, 0],
    }

    return pd.DataFrame(data)


def test_T11_rejet_valeur_manquante(valid_transactions_T11):
    """Une valeur NaN doit être rejetée."""

    df = valid_transactions_T11.copy()
    df.loc[0, "Amount"] = np.nan

    with pytest.raises(ValueError):
        validate_data(df)


def test_T11_rejet_valeur_infinie(valid_transactions_T11):
    """Une valeur infinie doit être rejetée."""

    df = valid_transactions_T11.copy()
    df.loc[0, "V1"] = np.inf

    with pytest.raises(ValueError):
        validate_data(df)


def test_T11_rejet_type_incorrect(valid_transactions_T11):
    """Une valeur textuelle doit être rejetée."""

    df = valid_transactions_T11.copy()
    df["Amount"] = df["Amount"].astype(object)
    df.loc[0, "Amount"] = "montant_invalide"

    with pytest.raises((TypeError, ValueError)):
        validate_data(df)


def test_T11_rejet_colonne_manquante(valid_transactions_T11):
    """Une variable obligatoire manquante doit être détectée."""

    df = valid_transactions_T11.drop(columns=["V14"])

    with pytest.raises(ValueError):
        validate_data(df)


def test_T11_rejet_cible_invalide(valid_transactions_T11):
    """La classe doit être exclusivement 0 ou 1."""

    df = valid_transactions_T11.copy()
    df.loc[0, "Class"] = 2

    with pytest.raises(ValueError):
        validate_data(df)


def test_T11_montant_extreme_fini(valid_transactions_T11):
    """
    Un montant très élevé mais fini ne doit pas être
    confondu avec une valeur techniquement invalide.
    """

    df = valid_transactions_T11.copy()
    df.loc[0, "Amount"] = 1_000_000.0

    assert validate_data(df) is True