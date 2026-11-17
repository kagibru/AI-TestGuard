
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(X, y, random_state=42):
    """
    Sépare les données en trois ensembles stratifiés :
    - 60 % entraînement ;
    - 20 % validation ;
    - 20 % test.
    """

    # Premier découpage : 60 % / 40 %
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )

    # Deuxième découpage : 20 % / 20 %
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


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
