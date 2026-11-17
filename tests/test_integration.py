
import numpy as np
import pandas as pd
import pytest

from app.preprocessing import preprocess_data, validate_data
from app.training import split_data, train_model


@pytest.mark.integration
def test_T05_pipeline_complet():
    """Vérifier le fonctionnement du pipeline sur les données Kaggle."""

    # 1. Charger les données réelles.
    df = pd.read_csv("data/creditcard.csv")

    # 2. Vérifier le contrat des données.
    assert validate_data(df) is True

    # 3. Préparer les variables.
    X, y = preprocess_data(df)

    assert X.shape[1] == 30
    assert len(X) == len(y)

    # 4. Séparer les données.
    X_train, X_val, X_test, y_train, _y_val, _y_test = split_data(X, y, random_state=42)

    # 5. Vérifier l'indépendance des partitions.
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    # 6. Entraîner uniquement sur Train.
    model = train_model(X_train, y_train)

    # 7. Produire des probabilités sur Validation.
    proba_val = model.predict_proba(X_val)[:, 1]

    assert len(proba_val) == len(X_val)
    assert np.isfinite(proba_val).all()
    assert ((proba_val >= 0) & (proba_val <= 1)).all()

    # 8. Vérifier que le scaler a été ajusté sur Train.
    scaler = model.named_steps["scaler"]

    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0).to_numpy(), rtol=1e-5, atol=1e-8,)

    # 9. Vérifier la conservation du taux de fraude
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)

    taux_global = y.mean()

    taux_train = y_train.mean()
    taux_val = y_val.mean()
    taux_test = y_test.mean()

    print(f"\nTaux global : {taux_global:.4%}")
    print(f"Taux Train  : {taux_train:.4%}")
    print(f"Taux Val    : {taux_val:.4%}")
    print(f"Taux Test   : {taux_test:.4%}")

    # Tolérance : 0,01 point de pourcentage.
    tolerance = 0.0001

    assert abs(taux_train - taux_global) <= tolerance
    assert abs(taux_val - taux_global) <= tolerance
    assert abs(taux_test - taux_global) <= tolerance


    for nom, cible in [
        ("Global", y),
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        print(
            f"{nom:12} | "
            f"Total : {len(cible):7} | "
            f"Fraudes : {(cible == 1).sum():4} | "
            f"Taux : {cible.mean():.4%}"
        )
