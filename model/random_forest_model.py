from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_model_and_params(random_state: int = 42):
    model = Pipeline([
        (
            "clf",
            RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }

    return model, param_grid
