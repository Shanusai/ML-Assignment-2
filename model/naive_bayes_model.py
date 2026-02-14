from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_and_params(random_state: int = 42):
    _ = random_state

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])

    param_grid = {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    }

    return model, param_grid
