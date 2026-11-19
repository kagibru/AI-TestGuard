import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.training import split_data, train_model

BASELINE_PATH = Path("reports/baseline_model.json")
COMPARISON_PATH = Path("reports/model_comparison.json")


def load_json(path):
    """Charge un fichier JSON de référence."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_T06_configuration_baseline():
    """Vérifie les paramètres essentiels de reproductibilité."""

    baseline = load_json(BASELINE_PATH)

    assert baseline["random_state"] == 42
    assert baseline["threshold"] == 0.90
    assert baseline["evaluation_set"] == "validation"

    split = baseline["split"]

    assert sum(split.values()) == pytest.approx(1.0)


def test_T06_entrainement_reproductible():
    """
    Deux entraînements identiques doivent produire
    les mêmes probabilités sur les mêmes données.
    """

    X = pd.DataFrame({
        "V1": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "Amount": [10, 20, 30, 100, 200, 300],
    })

    y = pd.Series([0, 0, 0, 1, 1, 1])

    model_1 = train_model(X, y)
    model_2 = train_model(X, y)

    probabilities_1 = model_1.predict_proba(X)
    probabilities_2 = model_2.predict_proba(X)

    np.testing.assert_allclose(
        probabilities_1,
        probabilities_2,
        rtol=0,
        atol=1e-12,
    )


def test_T06_partitionnement_reproductible():
    """
    Deux partitionnements avec la même graine
    doivent produire les mêmes indices.
    """

    X = pd.DataFrame({
        "V1": range(100),
    })

    y = pd.Series([0] * 90 + [1] * 10)

    result_1 = split_data(X, y, random_state=42)
    result_2 = split_data(X, y, random_state=42)

    for part_1, part_2 in zip(result_1, result_2):
        assert part_1.index.tolist() == part_2.index.tolist()


def test_T10_non_regression_performances():
    """
    Compare les métriques du modèle candidat
    avec les tolérances de la baseline.
    """

    baseline = load_json(BASELINE_PATH)
    results = load_json(COMPARISON_PATH)

    # Sélectionner la régression logistique
    # au seuil de classification 0.90.
    candidates = [
        result
        for result in results
        if result["model"] == baseline["model_name"]
        and np.isclose(
            result["threshold"],
            baseline["threshold"],
        )
    ]

    assert len(candidates) == 1, (
        "Configuration candidate introuvable ou ambiguë."
    )

    current = candidates[0]

    reference = baseline["metrics"]
    tolerances = baseline["tolerances"]

    # Métriques dont une diminution est défavorable.
    for metric in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        minimum = (
            reference[metric]
            - tolerances[f"{metric}_drop"]
        )

        assert current[metric] >= minimum, (
            f"Régression détectée sur {metric} : "
            f"{current[metric]:.4f} < {minimum:.4f}"
        )

    # Faux positifs : une augmentation est défavorable.
    max_fp = (
        reference["fp"]
        + tolerances["fp_increase"]
    )

    assert current["fp"] <= max_fp, (
        f"Trop de faux positifs : {current['fp']}"
    )

    # Faux négatifs : une augmentation est défavorable.
    max_fn = (
        reference["fn"]
        + tolerances["fn_increase"]
    )

    assert current["fn"] <= max_fn, (
        f"Trop de faux négatifs : {current['fn']}"
    )
