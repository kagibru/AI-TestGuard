import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.persistence import load_model, save_model
from app.training import train_model


@pytest.fixture
def trained_pipeline():
    """
    Fournit un petit pipeline entraîné.

    Le jeu synthétique permet de tester
    la persistance sans charger Kaggle.
    """

    X = pd.DataFrame({
        "V1": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "Amount": [10, 20, 30, 100, 200, 300],
    })

    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = train_model(X, y)

    return model, X


def test_T09_sauvegarde_pipeline(
    trained_pipeline,
    tmp_path,
):
    """Vérifie que le fichier modèle est créé."""

    model, _ = trained_pipeline

    filepath = tmp_path / "models" / "fraud_model.joblib"

    result = save_model(model, filepath)

    assert filepath.is_file()
    assert result == filepath


def test_T09_predictions_identiques_apres_rechargement(
    trained_pipeline,
    tmp_path,
):
    """
    Vérifie la conservation des probabilités
    et des prédictions après rechargement.
    """

    model, X = trained_pipeline

    filepath = tmp_path / "fraud_model.joblib"

    # Résultats avant sauvegarde.
    probabilities_before = model.predict_proba(X)
    predictions_before = model.predict(X)

    # Sauvegarder le pipeline.
    save_model(model, filepath)

    # Recharger le pipeline.
    loaded_model = load_model(filepath)

    # Résultats après rechargement.
    probabilities_after = loaded_model.predict_proba(X)
    predictions_after = loaded_model.predict(X)

    # Vérifier les probabilités.
    np.testing.assert_allclose(
        probabilities_before,
        probabilities_after,
        rtol=0,
        atol=1e-12,
    )

    # Vérifier les décisions.
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )

    # Vérifier les paramètres appris du scaler.
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_,
        loaded_model.named_steps["scaler"].mean_,
        rtol=0,
        atol=0,
    )

    # Vérifier les coefficients du classificateur.
    np.testing.assert_allclose(
        model.named_steps["classifier"].coef_,
        loaded_model.named_steps["classifier"].coef_,
        rtol=0,
        atol=0,
    )


def test_T09_fichier_inexistant(tmp_path):
    """Le chargement doit échouer si le fichier est absent."""

    filepath = tmp_path / "absent.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(filepath)


def test_T09_refus_objet_non_pipeline(tmp_path):
    """La sauvegarde doit refuser un objet non conforme."""

    filepath = tmp_path / "invalide.joblib"

    with pytest.raises(TypeError):
        save_model({"model": "invalide"}, filepath)

def test_T09_refus_sauvegarde_pipeline_non_entraine(tmp_path):
    """Un pipeline non entraîné ne doit pas être sauvegardé."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    filepath = tmp_path / "non_entraine.joblib"

    with pytest.raises(ValueError):
        save_model(model, filepath)

    assert not filepath.exists()


def test_T09_refus_chargement_objet_non_pipeline(tmp_path):
    """Le chargement doit rejeter un objet non conforme."""

    filepath = tmp_path / "objet_invalide.joblib"

    joblib.dump({"type": "invalide"}, filepath)

    with pytest.raises(TypeError):
        load_model(filepath)


def test_T09_refus_chargement_pipeline_non_entraine(tmp_path):
    """Le chargement doit rejeter un pipeline non entraîné."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    filepath = tmp_path / "pipeline_non_entraine.joblib"

    joblib.dump(model, filepath)

    with pytest.raises(ValueError):
        load_model(filepath)

