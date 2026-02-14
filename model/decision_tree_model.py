from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def get_model_and_params(random_state: int = 42):
    model = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=random_state)),
    ])

    param_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    return model, param_grid
