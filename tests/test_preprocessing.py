import numpy as np
import pandas as pd
import pytest

from app.preprocessing import preprocess_data, validate_data


@pytest.fixture
def donnees_valides():
    """Construire un petit jeu de données conforme au contrat."""
    donnees={
        "Time":[0.0, 1.0, 2.0],
        "Amount":[100.0, 200.0, 300.0],
        "Class":[0, 1, 0],
    }

    for i in range(1,29):
        donnees[f"V{i}"]=[0.1*i, 0.2*i, 0.3*i]
    return pd.DataFrame(donnees)


def test_T01_colonnes_requises_presentes(donnees_valides):
    """Un jeu de données conforme doit être accepté."""
    validate_data(donnees_valides)


def test_T01_colonne_obligatoire_manquante(donnees_valides):
    """Une colonne obligatoire absente doit être détectée."""
    donnees = donnees_valides.drop(columns=["V12"])

    with pytest.raises(ValueError):
        validate_data(donnees)

def test_T02_types_donnees_conformes(donnees_valides):
    """Les colonnes numériques doivent avoir un type numérique."""
    donnees = donnees_valides.copy()
    donnees["Amount"] = ["invalide", "250", "50"]

    with pytest.raises((ValueError, TypeError)):
        validate_data(donnees)


def test_T08_cible_binaire(donnees_valides):
    """La variable Class doit contenir uniquement 0 et 1."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "Class"] = 2

    with pytest.raises(ValueError):
        validate_data(donnees)


def test_T03_valeurs_non_finies(donnees_valides):
    """Une valeur infinie ne doit pas passer silencieusement."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "Amount"] = np.inf

    with pytest.raises(ValueError):
        validate_data(donnees)


def test_T01_colonnes_dupliquees(donnees_valides):
    """T01 : rejeter les noms de colonnes dupliqués."""
    donnees = donnees_valides.copy()

    colonnes = list(donnees.columns)
    colonnes[colonnes.index("V12")] = "V11"
    donnees.columns = colonnes

    with pytest.raises(ValueError):
        validate_data(donnees)


def test_T02_entree_non_dataframe():
    """T02 : rejeter un objet qui n'est pas un DataFrame."""
    with pytest.raises(TypeError):
        validate_data([1, 2, 3])


def test_T02_dataframe_vide():
    """Rejeter un DataFrame vide."""
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame())


def test_T03_valeur_nan(donnees_valides):
    """Rejeter une valeur manquante."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "Amount"] = np.nan

    with pytest.raises(ValueError):
        validate_data(donnees)


def test_T08_cible_negative(donnees_valides):
    """T08 : rejeter une cible égale à -1."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "Class"] = -1

    with pytest.raises(ValueError):
        validate_data(donnees)


def test_T03_pretraitement_valide(donnees_valides):
    """Le prétraitement doit produire X et y conformes."""
    X, y = preprocess_data(donnees_valides)

    assert X.shape == (3, 30)
    assert y.shape == (3,)
    assert "Class" not in X.columns
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_T03_pretraitement_rejette_nan(donnees_valides):
    """Le prétraitement doit rejeter les valeurs NaN."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "V1"] = np.nan

    with pytest.raises(ValueError):
        preprocess_data(donnees)


def test_T03_pretraitement_rejette_inf(donnees_valides):
    """Le prétraitement doit rejeter les valeurs infinies."""
    donnees = donnees_valides.copy()
    donnees.loc[0, "Amount"] = np.inf

    with pytest.raises(ValueError):
        preprocess_data(donnees)


def test_T03_pretraitement_conserve_cible(donnees_valides):
    """Les étiquettes doivent rester inchangées."""
    X, y = preprocess_data(donnees_valides)

    assert y.tolist() == donnees_valides["Class"].tolist()
    assert X.columns.tolist() == [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ]
