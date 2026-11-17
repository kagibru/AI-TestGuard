import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (["Time"] + [f"V{i}" for i in range(1, 29)]+ ["Amount", "Class"])


def validate_data(df: pd.DataFrame) -> bool:
    """
    Vérifie la conformité d'un jeu de données bancaires.

    Contrôles :
    - T01 : présence et unicité des colonnes obligatoires ;
    - T02 : conformité des types numériques ;
    - T08 : validité de la cible binaire Class ;
    - absence de valeurs manquantes ou infinies ;
    - absence de jeu de données vide.

    Returns:
        True si les données sont conformes.

    Raises:
        TypeError: si l'objet ou les types sont invalides.
        ValueError: si les données ne respectent pas le contrat.
    """

    # Vérifier que l'entrée est un DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas.")

    # Vérifier que le jeu de données n'est pas vide.
    if df.empty:
        raise ValueError("Le jeu de données est vide.")

    # T01 : vérifier l'unicité des noms de colonnes.
    if not df.columns.is_unique:
        raise ValueError("Des noms de colonnes sont dupliqués.")

    # T01 : vérifier la présence des colonnes obligatoires.
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires manquantes : {missing_columns}"
        )

    # T02 : vérifier les types numériques.
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"La colonne {col} doit être numérique."
            )

    # Vérifier l'absence de valeurs manquantes.
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Le jeu de données contient des valeurs manquantes."
        )

    # Vérifier l'absence de valeurs infinies.
    if not np.isfinite(
        df[REQUIRED_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Le jeu de données contient des valeurs non finies."
        )

    # T08 : vérifier que la cible est strictement binaire.
    if not df["Class"].isin([0, 1]).all():
        raise ValueError(
            "La variable Class doit contenir uniquement 0 ou 1."
        )

    return True
