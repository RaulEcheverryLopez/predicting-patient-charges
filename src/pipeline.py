import pandas as pd
import numpy as np
from pycaret.regression import *
import joblib
import os

class MLPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = 'ml_models/model.joblib'
        self.setup_path = 'ml_models/setup.joblib'
        
    def load_data(self, filepath=None):
        """Load data from CSV file or use PyCaret's built-in dataset"""
        if filepath:
            return pd.read_csv(filepath)
        else:
            from pycaret.datasets import get_data
            return get_data('insurance')
    
    def setup_pipeline(self, data):
        """Setup PyCaret environment"""
        setup(data, 
              target='charges',
              session_id=123,
              normalize=True,
              polynomial_features=True,
              trigonometry_features=True,
              feature_interaction=True,
              bin_numeric_features=['age', 'bmi'],
              silent=True)
        
        # Save setup for later use
        save_config(self.setup_path)
    
    def train_model(self, data=None):
        """Train the model using PyCaret"""
        if data is None:
            data = self.load_data()
            
        # Setup environment
        self.setup_pipeline(data)
        
        # Compare models
        best_model = compare_models()
        
        # Finalize model
        self.model = finalize_model(best_model)
        
        # Save model
        self.save_model()
        
        return self.model
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path) and os.path.exists(self.setup_path):
            # Load setup
            load_config(self.setup_path)
            
            # Load model
            self.model = joblib.load(self.model_path)
            return self.model
        else:
            raise FileNotFoundError("Model or setup files not found. Please train the model first.")
    
    def predict(self, features):
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
            
        # Convert features to DataFrame
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=['age', 'bmi', 'children', 'smoker', 'sex', 'region'])
            
        # Make prediction
        prediction = predict_model(self.model, data=features)
        return prediction['prediction_label'].values[0] 