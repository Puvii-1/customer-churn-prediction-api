import pandas as pd
from src.data import generate_churn_data
from src.train import build_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_generate_churn_data_shape():
    df = generate_churn_data(n_customers=500)
    assert len(df) == 500
    assert "churned" in df.columns
    assert set(df["churned"].unique()) <= {0, 1}


def test_churn_rate_is_realistic():
    df = generate_churn_data(n_customers=5000)
    rate = df["churned"].mean()
    # A real-world churn rate is typically 15-35%; assert we're in a
    # believable range rather than the data being degenerate (all 0s/1s
    # or wildly unrealistic like 80%+)
    assert 0.10 < rate < 0.40


def test_pipeline_trains_and_predicts():
    df = generate_churn_data(n_customers=500)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]

    pipeline = build_pipeline(n_estimators=20, max_depth=4)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert all(0.0 <= p <= 1.0 for p in probabilities[:, 1])


def test_pipeline_handles_single_row_prediction():
    df = generate_churn_data(n_customers=500)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]

    pipeline = build_pipeline(n_estimators=20, max_depth=4)
    pipeline.fit(X, y)

    single_row = X.iloc[[0]]
    prediction = pipeline.predict(single_row)
    assert len(prediction) == 1