
import numpy as np
import pytest

from app.evaluation import evaluate_model


def test_T05_evaluation_metriques():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.85]

    metrics = evaluate_model(y_true, y_pred, y_proba)

    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 3

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.75)

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_T05_longueurs_incompatibles():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 1],
            [0],
            [0.2, 0.8],
        )


def test_T05_probabilites_invalides():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 1],
            [0, 1],
            [0.2, 1.5],
        )


def test_T05_classes_invalides():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 2],
            [0, 1],
            [0.2, 0.8],
        )


def test_T05_probabilites_non_finies():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 1],
            [0, 1],
            [0.2, np.nan],
        )


def test_T05_evaluation_sans_deux_classes():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 0, 0],
            [0, 0, 0],
            [0.1, 0.2, 0.3],
        )


def test_T05_entrees_non_unidimensionnelles():
    with pytest.raises(ValueError):
        evaluate_model(
            [[0], [1]],
            [0, 1],
            [0.2, 0.8],
        )


def test_T05_jeu_evaluation_vide():
    with pytest.raises(ValueError):
        evaluate_model([], [], [])


def test_T05_predictions_invalides():
    with pytest.raises(ValueError):
        evaluate_model(
            [0, 1],
            [0, 2],
            [0.2, 0.8],
        )
