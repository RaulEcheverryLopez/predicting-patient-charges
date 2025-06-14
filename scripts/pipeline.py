import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from pycaret.datasets import get_data
from pycaret.regression import (
    compare_models,
    finalize_model,
    load_model,
    predict_model,
    save_model,
    setup,
)

from config import *


class MLPipeline:
    def __init__(self):
        self.model = None
        self.setup = None
        self.model_path = MODEL_PATH

    def load_data(self, file_path=None):
        """Load data from file or use PyCaret's built-in dataset"""
        if file_path:
            return pd.read_csv(file_path)
        return get_data("insurance")

    def setup_pipeline(self, data):
        """Configure PyCaret environment"""
        self.setup = setup(
            data=data,
            target="charges",
            session_id=123,
            normalize=True,
            polynomial_features=True,
            bin_numeric_features=["age", "bmi"],
        )

    def train_model(self):
        """Train the model using PyCaret"""
        if not self.setup:
            raise ValueError("Setup must be run before training")

        # Compare models and select the best one
        best_model = compare_models()

        # Finalize the model
        self.model = finalize_model(best_model)

        # Save the model
        self.save_model()

    def save_model(self):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        save_model(self.model, str(self.model_path))

    def load_model(self):
        """Load the model from disk"""
        try:
            self.model = load_model(str(self.model_path))
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, features):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Convert features to DataFrame
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Make predictions
        predictions = predict_model(self.model, data=features)
        return predictions["prediction_label"].values[0]


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLPipeline()

    # Load data
    data = pipeline.load_data()

    # Setup environment
    pipeline.setup_pipeline(data)

    # Train model
    pipeline.train_model()

    print("Model training completed successfully!")
