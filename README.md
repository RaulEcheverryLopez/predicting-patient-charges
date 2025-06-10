# Patient Charges Predictor

A machine learning project for predicting patient medical charges using PyCaret, FastAPI, and Streamlit.

## Project Structure

```
patient-charges-predictor/
├── data/
│   ├── raw/          # Raw data files
│   └── processed/    # Processed data files
├── models/           # Trained models
├── notebooks/        # Jupyter notebooks
├── scripts/
│   ├── app.py        # FastAPI application
│   ├── pipeline.py   # ML pipeline
│   └── streamlit_app.py  # Streamlit application
├── .github/
│   └── workflows/    # GitHub Actions workflows
├── .streamlit/       # Streamlit configuration
├── config.py         # Project configuration
├── Dockerfile        # Docker configuration
└── requirements.txt  # Python dependencies
```

## Features

- Machine learning pipeline using PyCaret
- FastAPI backend for predictions
- Streamlit web interface
- Docker containerization
- CI/CD with GitHub Actions
- Data analysis and visualization
- Model retraining automation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/patient-charges-predictor.git
cd patient-charges-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. Train the model:
```bash
python scripts/pipeline.py
```

2. Run the FastAPI application:
```bash
uvicorn scripts.app:app --host 0.0.0.0 --port 8000
```

3. Run the Streamlit application:
```bash
streamlit run scripts/streamlit_app.py
```

### Docker

1. Build the Docker image:
```bash
docker build -t predictorcharges.azurecr.io/predictor-charges:latest .
docker push predictorcharges.azurecr.io/predictor-charges:latest
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 patient-charges-predictor
```

## API Documentation

Once the FastAPI application is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example API Request

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "age": 30,
    "sex": "male",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast"
}

response = requests.post(url, json=data)
print(response.json())
```

## Web Interface

The Streamlit web interface is available at http://localhost:8501

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. **Test**: Runs unit tests and code coverage
2. **Lint**: Checks code style and formatting
3. **Train**: Trains the model if data has changed
4. **Build and Push**: Builds and pushes Docker image to DockerHub

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.