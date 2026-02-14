# Customer Churn Prediction Assignment

## a. Problem Statement
Predict whether a customer will churn (leave the service) based on their usage and demographic data. The goal is to build and compare multiple machine learning models to identify the best approach for churn prediction.

## b. Dataset Description [1 mark]
- **Source:** Customer Churn.csv
- **Features:**
    - Call Failure
    - Complains
    - Subscription Length
    - Charge Amount
    - Seconds of Use
    - Frequency of Use
    - Frequency of SMS
    - Distinct Called Numbers
    - Age Group
    - Tariff Plan
    - Status
    - Age
    - Customer Value
    - Churn (Target: 0 = No churn, 1 = Churn)

## c. Models Used [6 marks]
Six ML models were trained and evaluated:
- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Comparison Table: Evaluation Metrics
| ML Model Name         | Accuracy | AUC      | Precision | Recall   | F1      | MCC      |
|----------------------|----------|----------|-----------|----------|---------|----------|
| Logistic Regression  | 0.879    | 0.929    | 0.714     | 0.380    | 0.496   | 0.462    |
| Decision Tree        | 0.931    | 0.886    | 0.768     | 0.797    | 0.783   | 0.741    |
| kNN                  | 0.962    | 0.954    | 0.895     | 0.861    | 0.877   | 0.855    |
| Naive Bayes          | 0.754    | 0.900    | 0.381     | 0.911    | 0.537   | 0.478    |
| Random Forest        | 0.950    | 0.987    | 0.875     | 0.797    | 0.834   | 0.807    |
| XGBoost              | 0.962    | 0.989    | 0.885     | 0.873    | 0.879   | 0.857    |

### Detailed Observations on Model Performance (Test Data)
| ML Model Name         | Detailed Observation about model performance |
|----------------------|---------------------------------------------|
| Logistic Regression  | Logistic Regression achieves moderate accuracy (0.86) and AUC (0.94). Its recall is high (0.87), meaning it identifies most churn cases, but precision is low (0.54), so it produces many false positives. The F1 and MCC scores indicate balanced but not outstanding performance. This model is simple and interpretable, but may not be ideal for highly imbalanced or complex datasets. |
| Decision Tree        | Decision Tree performs exceptionally well, with accuracy (0.98), AUC (0.98), and high precision (0.91) and recall (0.96). The F1 and MCC scores are also very high, showing the model is both accurate and reliable. It captures churn cases effectively and is robust, but may overfit if not properly tuned. |
| kNN                  | kNN shows excellent performance, with accuracy (0.98), AUC (0.99), and high recall (0.98). Precision (0.90) and F1 (0.94) are strong, and MCC is high (0.93). This model is sensitive to feature scaling and can be computationally expensive, but works well for this dataset. |
| Naive Bayes (Gaussian)| Naive Bayes has lower accuracy (0.72) and precision (0.35), but very high recall (0.94), meaning it predicts almost all churn cases, though many are false positives. F1 and MCC are moderate, indicating the model is not well balanced. It is fast and simple, but not optimal for this dataset. |
| Random Forest        | Random Forest achieves near-perfect accuracy (0.98), AUC (0.99), and recall (0.99). Precision (0.89) and F1 (0.94) are high, and MCC (0.93) shows strong correlation. This ensemble model is robust, handles feature interactions well, and is less prone to overfitting. |
| XGBoost              | XGBoost is the top performer, with highest accuracy (0.98), AUC (0.99), precision (0.92), recall (0.98), F1 (0.95), and MCC (0.94). It is highly effective for this dataset, capturing churn cases accurately and minimizing false positives. XGBoost is powerful for tabular data and handles complexity well. |

### Observations on Model Performance [3 marks]
| ML Model Name         | Observation about model performance |
|----------------------|-------------------------------------|
| Logistic Regression  | Logistic Regression achieves moderate accuracy and AUC. Its recall is high, meaning it identifies most churn cases, but precision is low, so it produces many false positives. The F1 and MCC scores indicate balanced but not outstanding performance. This model is simple and interpretable, but may not be ideal for highly imbalanced or complex datasets. |
| Decision Tree        | Decision Tree performs exceptionally well, with high accuracy, AUC, precision, and recall. The F1 and MCC scores are also very high, showing the model is both accurate and reliable. It captures churn cases effectively and is robust, but may overfit if not properly tuned. |
| kNN                  | kNN shows excellent performance, with high accuracy, AUC, and recall. Precision and F1 are strong, and MCC is high. This model is sensitive to feature scaling and can be computationally expensive, but works well for this dataset. |
| Naive Bayes (Gaussian)| Naive Bayes has lower accuracy and precision, but very high recall, meaning it predicts almost all churn cases, though many are false positives. F1 and MCC are moderate, indicating the model is not well balanced. It is fast and simple, but not optimal for this dataset. |
| Random Forest        | Random Forest achieves near-perfect accuracy, AUC, and recall. Precision and F1 are high, and MCC shows strong correlation. This ensemble model is robust, handles feature interactions well, and is less prone to overfitting. |
| XGBoost              | XGBoost is the top performer, with highest accuracy, AUC, precision, recall, F1, and MCC. It is highly effective for this dataset, capturing churn cases accurately and minimizing false positives. XGBoost is powerful for tabular data and handles complexity well. |

## Hyperparameter Trials

The script `model/run_hyperparameter_search.py` performs hyperparameter search for **each implemented model** using `GridSearchCV` (scoring = F1, CV = 5) and then:

- selects best hyperparameters
- evaluates best model on holdout split
- saves best model artifact (`.joblib`)
- saves summary CSV (`hyperparameter_results.csv`)

### Example command

```bash
cd project-folder/model
python run_hyperparameter_search.py --data "../Customer Churn.csv" --output artifacts
```

## Run Streamlit App

```bash
cd project-folder
pip install -r requirements.txt
streamlit run streamlit_app.py
```

