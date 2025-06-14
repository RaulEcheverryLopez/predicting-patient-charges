import numpy as np
import pandas as pd
import pytest

from src.pipeline import MLPipeline


def test_pipeline_initialization():
    pipeline = MLPipeline()
    assert pipeline.model is None
    assert pipeline.scaler is not None


def test_preprocess_data():
    pipeline = MLPipeline()
    # Create sample data
    data = {
        "age": [30, 40, 50],
        "bmi": [25, 30, 35],
        "children": [0, 1, 2],
        "smoker": [0, 1, 0],
        "sex": [1, 0, 1],
        "region": [0, 1, 2],
        "charges": [1000, 2000, 3000],
    }
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
