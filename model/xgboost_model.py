from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_model_and_params(random_state: int = 42):

    model = Pipeline([
        (
            "clf",
            XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.2],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    return model, param_grid
