
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y):
    """
    Entraîne un modèle de régression logistique.

    Paramètres :
        X : variables explicatives d'entraînement.
        y : étiquettes d'entraînement (0 ou 1).

    Retour :
        Pipeline scikit-learn entraîné.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model
