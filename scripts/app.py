import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import API_HOST, API_PORT
from pipeline import MLPipeline

app = FastAPI(
    title="Patient Charges Prediction API",
    description="API for predicting patient medical charges using PyCaret",
    version="1.0.0",
)

# Initialize pipeline
pipeline = MLPipeline()


# Load model at startup
@app.on_event("startup")
async def startup_event():
    if not pipeline.load_model():
        print("Model not found. Training new model...")
        try:
            data = pipeline.load_data()
            pipeline.setup_pipeline(data)
            pipeline.train_model()
            print("Model trained successfully!")
        except Exception as e:
            print(f"Error training model: {str(e)}")


class PatientData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.post("/predict")
async def predict(patient_data: PatientData):
    if pipeline.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to dictionary
        features = patient_data.dict()

        # Make prediction
        prediction = pipeline.predict(features)

        return {"prediction": float(prediction), "input_data": features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
