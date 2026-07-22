"""
Trains a churn classifier and logs everything to MLflow: parameters,
metrics, the model artifact itself, and a confusion matrix plot.

Run with: python -m src.train
Then view results with: mlflow ui   (opens at http://localhost:5000)
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the plot to a file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.data import generate_churn_data

MODEL_PATH = "models/churn_model.joblib"
CATEGORICAL_FEATURES = ["contract_type"]
NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets", "is_senior_citizen", "has_tech_support", "payment_delay_days",
]


def build_pipeline(n_estimators: int, max_depth: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through unchanged
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight="balanced"
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train(n_estimators: int = 200, max_depth: int = 8):
    mlflow.set_experiment("churn-prediction")

    df = generate_churn_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        pipeline = build_pipeline(n_estimators, max_depth)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Confusion matrix as an artifact — useful to eyeball in the MLflow UI
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close(fig)

        mlflow.sklearn.log_model(pipeline, name="model")

        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(pipeline, MODEL_PATH)

        print("Training complete.")
        for name, value in metrics.items():
            print(f"  {name}: {value:.3f}")
        print(f"Model saved to {MODEL_PATH}")

    return pipeline, metrics


if __name__ == "__main__":
    train()