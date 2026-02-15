import io
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, auc
)

st.set_page_config(page_title="Customer Churn – ML Assignment 2", layout="wide")
st.title("Customer Churn – ML Assignment 2")

TARGET_COL = "Churn"

def _safe_auc(y_true, y_prob):
    try:
        if y_prob is None or len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

def evaluate(model, X, y=None, threshold=0.5):
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    metrics = None
    if y is not None:
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": _safe_auc(y, y_prob),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred)
        }
    return metrics, y_pred, y_prob

def show_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

def show_roc_curve(y_true, y_prob, title="ROC Curve"):
    if y_prob is None or len(np.unique(y_true)) < 2:
        st.info("ROC curve not available (need probabilities and both classes in ground truth).")
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.header("1.1 Exploratory Data Analysis (EDA)")
st.markdown("""
EDA was performed on the dataset (`Customer Churn.csv`).

- **Dataset overview:** 3,150 samples and 13 input features (14 columns including the `Churn` target).

- **Class imbalance:** `Churn` is the minority class — churn rate ≈ 15.71% (0.1571).

- **Top numeric correlations with `Churn` (absolute Pearson):**
    - `Complains`: 0.532
    - `Status`: 0.499
    - `Frequency of use`: 0.303
    - `Seconds of Use`: 0.299
    - `Customer Value`: 0.289
    - `Distinct Called Numbers`: 0.279
    - `Frequency of SMS`: 0.221
    - `Charge  Amount`: 0.202
    - `Tariff Plan`: 0.106
    - `Subscription  Length`: 0.033

Saved visuals (`eda_churn_distribution.png`, `eda_corr_heatmap.png`) are shown.
            
Best metric to take into account for this imbalanced classification problem is **MCC** (Matthews Correlation Coefficient) as it considers all four confusion matrix categories and is robust to class imbalance.
""")

if os.path.exists("eda_churn_distribution.png") and os.path.exists("eda_corr_heatmap.png"):
    c1, c2 = st.columns(2)
    with c1:
        st.image("eda_churn_distribution.png", caption="Churn Class Distribution", width=500)
    with c2:
        st.image("eda_corr_heatmap.png", caption="Feature Correlation Heatmap", width=500)
else:
    st.info("EDA plots not found. Please run the EDA script to generate them.")

# Brief description of preprocessing & split performed elsewhere
st.header("1.2 Preprocessing & Split — Description")
st.markdown(
    """
    The dataset is preprocessed and split using the following steps:

    - Normalize column names (strip and collapse whitespace).
    - Trim whitespace from string/object columns.
    - Added log1p-transformed copies for identified skewed numeric features (e.g., `Charge Amount`, `Customer Value`, `Seconds of Use`).
    - Ensured the `Churn` target is integer-typed and perform a stratified train/test split (80/20) to preserve class balance; outputs saved as `train.csv` and `test.csv`.
    """
)

st.markdown("---")
st.header("2. Model Selection and Prediction")
st.markdown("""
All models are pre-trained on the training set. Please select a model to make predictions on the uploaded test file.
""")

st.sidebar.title("Controls")
threshold = 0.2
model_choice = st.sidebar.selectbox(
    "Choose a model",
    [
        "All models",
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes (Gaussian)",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded = st.sidebar.file_uploader(
    "Upload TEST CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.header("Download Sample Test Data")
with open("test.csv", "rb") as f:
    st.sidebar.download_button(
        label="Customer Churn Test CSV",
        data=f,
        file_name="customer_churn_test.csv",
        mime="text/csv"
    )

@st.cache_resource(show_spinner=False)
def load_models():
    model_files = {
        "Logistic Regression": "model/artifacts/logistic_regression_best.joblib",
        "Decision Tree": "model/artifacts/decision_tree_best.joblib",
        "kNN": "model/artifacts/knn_best.joblib",
        "Naive Bayes (Gaussian)": "model/artifacts/naive_bayes_best.joblib",
        "Random Forest": "model/artifacts/random_forest_best.joblib",
        "XGBoost": "model/artifacts/xgboost_best.joblib",
    }
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

fitted_models = load_models()
if not fitted_models:
    st.error("No pre-trained models found. Please run the training script first.")
    st.stop()

if os.path.exists("model/artifacts/hyperparameter_results.csv"):
    st.subheader("Model Leaderboard (Validation Performance)")
    leaderboard = pd.read_csv("model/artifacts/hyperparameter_results.csv")
    leaderboard = leaderboard.rename(columns={"model": "Model"})
    leaderboard["Model"] = leaderboard["Model"].str.replace("_", " ").str.title()
    leaderboard = leaderboard.set_index("Model")
    st.dataframe(leaderboard[["accuracy", "auc", "precision", "recall", "f1", "mcc"]], use_container_width=True)
else:
    st.info("Leaderboard not found. Please run the training script.")

st.header("3. Evaluate on Uploaded TEST CSV")
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)

        y_test = None
        if TARGET_COL in test_df.columns:
            y_test = test_df[TARGET_COL].astype(int).values
            X_test = test_df.drop(columns=[TARGET_COL])
        else:
            X_test = test_df

        model_feature_names = list(fitted_models.values())[0].feature_names_in_
        missing_cols = [c for c in model_feature_names if c not in X_test.columns]
        extra_cols = [c for c in X_test.columns if c not in model_feature_names]
        if missing_cols:
            st.warning(f"Missing columns in uploaded file (filled with 0): {missing_cols}")
        if extra_cols:
            st.info(f"Extra columns will be ignored: {extra_cols}")
        X_test = X_test.reindex(columns=model_feature_names, fill_value=0)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        st.subheader("Results on Uploaded Test")
        if model_choice == "All models":
            rows = []
            for name, model in fitted_models.items():
                metrics, y_pred, y_prob = evaluate(model, X_test, y_test, threshold=threshold)
                row = {"Model": name}
                if metrics is not None:
                    row.update(metrics)
                rows.append(row)
            test_metrics_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(test_metrics_df, use_container_width=True)

        else:
            if model_choice not in fitted_models:
                st.error("Selected model is unavailable.")
            else:
                model = fitted_models[model_choice]
                metrics, y_pred, y_prob = evaluate(model, X_test, y_test, threshold=threshold)

                if y_test is not None:
                    st.write("**Metrics**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    with c2:
                        auc_val = metrics["AUC"]
                        st.metric("AUC", "N/A" if (auc_val is None or np.isnan(auc_val)) else f"{auc_val:.4f}")
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    with c3:
                        st.metric("F1", f"{metrics['F1']:.4f}")
                        st.metric("MCC", f"{metrics['MCC']:.4f}")

                    show_confusion_matrix(y_test, y_pred, title=f"{model_choice} – Confusion Matrix")

                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
                    st.code(report, language="text")

                    st.subheader("ROC Curve")
                    show_roc_curve(y_test, y_prob, title=f"{model_choice} – ROC Curve")

                else:
                    st.info("No ground truth in uploaded file; showing predictions only.")
                    out = pd.DataFrame({"Predicted": y_pred})
                    if y_prob is not None:
                        out["Probability"] = y_prob
                    st.dataframe(out.head(50), use_container_width=True)

                    pbuf = io.StringIO()
                    out.to_csv(pbuf, index=False)

    except Exception as e:
        st.error(f"Could not evaluate on uploaded file: {e}")
else:
    st.info("Upload a small TEST CSV in the sidebar to evaluate models on the data")

