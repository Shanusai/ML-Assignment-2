import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from decision_tree_model import get_model_and_params as dt
from knn_model import get_model_and_params as knn
from logistic_regression_model import get_model_and_params as lr
from naive_bayes_model import get_model_and_params as nb
from random_forest_model import get_model_and_params as rf
from xgboost_model import get_model_and_params as xgb

TARGET_COL = "Churn"


def safe_auc(y_true, y_prob):
    if y_prob is None or len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def evaluate(model, x_test, y_test):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = None
        y_pred = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": safe_auc(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }


def run_search(data_path: Path, output_dir: Path, test_size: float, random_state: int):
    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {data_path}")

    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model_builders = {
        "logistic_regression": lr,
        "decision_tree": dt,
        "knn": knn,
        "naive_bayes": nb,
        "random_forest": rf,
        "xgboost": xgb,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name, builder in model_builders.items():
        model, param_grid = builder(random_state=random_state)

        if model is None:
            print(f"Skipping {model_name}: dependency not available")
            continue

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        holdout_metrics = evaluate(best_model, x_test, y_test)

        artifact_path = output_dir / f"{model_name}_best.joblib"
        joblib.dump(best_model, artifact_path)

        rows.append(
            {
                "model": model_name,
                "best_params": search.best_params_,
                "best_cv_f1": search.best_score_,
                **holdout_metrics,
                "artifact": str(artifact_path),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False)
    results_path = output_dir / "hyperparameter_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"Saved results: {results_path}")
    print(results_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for all assignment models.")
    parser.add_argument(
        "--data",
        default="../Customer Churn.csv",
        help="Path to training data CSV with target column 'Churn'.",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory to save best model artifacts and metrics CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        test_size=args.test_size,
        random_state=args.random_state,
    )
