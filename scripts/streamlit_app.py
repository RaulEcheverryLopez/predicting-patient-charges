import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import MLPipeline
from config import STREAMLIT_HOST, STREAMLIT_PORT

# Set page config
st.set_page_config(
    page_title="Patient Charges Predictor",
    page_icon="🏥",
    layout="wide"
)

# Initialize pipeline
pipeline = MLPipeline()

# Load model
if not pipeline.load_model():
    st.error("Model not loaded. Please train the model first.")
    st.stop()

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=100)
    st.title("Patient Charges Predictor")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Prediction", "Data Analysis", "About"],
        icons=["calculator", "graph-up", "info-circle"],
        default_index=0
    )

# Prediction page
if selected == "Prediction":
    st.header("Predict Medical Charges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    if st.button("Predict"):
        # Prepare input data
        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
        
        # Make prediction
        try:
            prediction = pipeline.predict(input_data)
            
            # Display prediction
            st.success(f"Predicted Medical Charges: ${prediction:,.2f}")
            
            # Display input data
            st.subheader("Input Data")
            st.json(input_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Data Analysis page
elif selected == "Data Analysis":
    st.header("Data Analysis")
    
    # Load data
    data = pipeline.load_data()
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())
    
    # Display data distribution
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig, ax = plt.subplots()
        sns.histplot(data=data, x="age", bins=30)
        plt.title("Age Distribution")
        st.pyplot(fig)
        
        # BMI distribution
        fig, ax = plt.subplots()
        sns.histplot(data=data, x="bmi", bins=30)
        plt.title("BMI Distribution")
        st.pyplot(fig)
    
    with col2:
        # Charges distribution
        fig, ax = plt.subplots()
        sns.histplot(data=data, x="charges", bins=30)
        plt.title("Charges Distribution")
        st.pyplot(fig)
        
        # Children distribution
        fig, ax = plt.subplots()
        sns.countplot(data=data, x="children")
        plt.title("Number of Children")
        st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

# About page
else:
    st.header("About")
    
    st.markdown("""
    ### Patient Charges Predictor
    
    This application uses machine learning to predict medical charges based on patient characteristics.
    
    #### Features
    - Age
    - Sex
    - BMI (Body Mass Index)
    - Number of Children
    - Smoking Status
    - Region
    
    #### Technology Stack
    - FastAPI for the backend API
    - Streamlit for the web interface
    - PyCaret for machine learning
    - Docker for containerization
    - GitHub Actions for CI/CD
    
    #### Data Source
    The model is trained on the Insurance dataset from PyCaret.
    """)
    
    st.info("For more information, please visit the GitHub repository.")

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__)),
        "--server.address", STREAMLIT_HOST,
        "--server.port", str(STREAMLIT_PORT)
    ]
    sys.exit(stcli.main()) 