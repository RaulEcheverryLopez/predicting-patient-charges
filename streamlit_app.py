import pandas as pd
import requests
import streamlit as st

from scripts.pipeline import MLPipeline

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Cargos Médicos", page_icon="🏥", layout="wide"
)

# Título y descripción
st.title("Predicción de Cargos Médicos")
st.markdown(
    """
Esta aplicación predice los cargos médicos basándose en las características del paciente.
Ingrese los detalles del paciente a continuación para obtener una predicción.
"""
)

# Inicializar el pipeline
pipeline = MLPipeline()

# Cargar el modelo
if not pipeline.load_model():
    st.warning("Modelo no encontrado. Entrenando nuevo modelo...")
    data = pipeline.load_data()
    pipeline.setup_pipeline(data)
    pipeline.train_model()
    st.success("Modelo entrenado exitosamente!")

# Crear el formulario de entrada
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1
        )
        children = st.number_input(
            "Número de Hijos", min_value=0, max_value=10, value=0
        )

    with col2:
        sex = st.selectbox("Género", ["male", "female"])
        smoker = st.selectbox("Fumador", ["yes", "no"])
        region = st.selectbox(
            "Región", ["southwest", "southeast", "northwest", "northeast"]
        )

    submit_button = st.form_submit_button("Predecir Cargos Médicos")

# Procesar la predicción cuando se envía el formulario
if submit_button:
    try:
        # Crear diccionario de características
        features = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }

        # Obtener predicción
        prediction = pipeline.predict(features)

        # Mostrar resultado
        st.success(f"Predicción de Cargos Médicos: ${prediction:,.2f}")

        # Mostrar detalles de entrada
        st.subheader("Detalles de Entrada")
        st.json(features)

    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")

# Información adicional
st.markdown("---")
st.markdown(
    """
### Acerca de la Aplicación
Esta aplicación utiliza un modelo de Machine Learning entrenado con PyCaret para predecir los cargos médicos.
El modelo fue entrenado con datos históricos de pacientes y considera las siguientes variables:
- Edad
- Género
- BMI (Índice de Masa Corporal)
- Número de Hijos
- Estado de Fumador
- Región
"""
)
