# Mock pipeline module for testing imports without dependencies

class MLPipeline:
    def __init__(self):
        self.model = None
        self.scaler = "mock_scaler"
    
    def preprocess_data(self, df):
        # Mock implementation for testing
        return ["X_train", "X_test"], ["y_train", "y_test"]