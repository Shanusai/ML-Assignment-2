from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_and_params(random_state: int = 42):
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                random_state=random_state,
                solver="lbfgs",
                penalty="l2",
                max_iter=1000,
            ),
        ),
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__max_iter": [500, 1000],
    }

    return model, param_grid
