
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba):
    """
    Évalue les performances d'un classificateur binaire.

    y_true : classes réelles (0 ou 1)
    y_pred : classes prédites (0 ou 1)
    y_proba : probabilités estimées de la classe 1

    Retourne un dictionnaire de métriques.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    # Vérifier les dimensions.
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_proba.ndim != 1:
        raise ValueError("Les entrées doivent être unidimensionnelles.")

    if len(y_true) == 0:
        raise ValueError("Le jeu d'évaluation est vide.")

    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("Les entrées doivent avoir la même longueur.")

    # Vérifier les classes et les probabilités.
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true doit contenir uniquement 0 et 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred doit contenir uniquement 0 et 1.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Les probabilités doivent être finies.")

    if not ((y_proba >= 0) & (y_proba <= 1)).all():
        raise ValueError("Les probabilités doivent être entre 0 et 1.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Les métriques ROC-AUC et AP exigent ici les deux classes.
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "L'évaluation ROC-AUC/AP exige les deux classes."
        )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score( y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
