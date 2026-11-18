
# FICHIER : app/persistence.py

"""
Fonctions de sauvegarde et de chargement
d'un pipeline de Machine Learning entraîné.
"""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model, filepath):
    """
    Sauvegarde un pipeline entraîné sur disque.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Pipeline contenant le prétraitement et le modèle.

    filepath : str ou Path
        Emplacement du fichier de sauvegarde.

    Returns
    -------
    Path
        Chemin du fichier créé.
    """

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Le modèle doit être un pipeline scikit-learn."
        )

    if not hasattr(model, "classes_"):
        raise ValueError(
            "Le pipeline doit être entraîné avant sauvegarde."
        )

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, filepath)

    return filepath


def load_model(filepath):
    """
    Recharge un pipeline précédemment sauvegardé.

    Attention : charger uniquement des fichiers fiables.
    """

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Fichier modèle introuvable : {filepath}"
        )

    model = joblib.load(filepath)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Le fichier ne contient pas un pipeline valide."
        )

    if not hasattr(model, "classes_"):
        raise ValueError(
            "Le pipeline chargé n'est pas entraîné."
        )

    return model
