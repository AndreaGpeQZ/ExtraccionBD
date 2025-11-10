# ============================================================
# Predicción de Readmisión por Diabetes
# ============================================================
# Autor: Quintana Zepeda Andrea Guadalupe
# Fecha: 09 de Noviembre del 2025
# Descripción:
# Aplicación web en Streamlit que utiliza un modelo entrenado (.pkl)
# para predecir si un paciente diabético será readmitido en el hospital.
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# Cargar modelo
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("diabetes.pkl")

model = load_model()

# ==============================
# Configuración de la página
# ==============================
st.set_page_config(
    page_title="Predicción de Readmisión por Diabetes",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Predicción de Readmisión por Diabetes")
st.markdown(
    """
    Esta aplicación utiliza un modelo de **Machine Learning**
    para predecir si un paciente diabético será readmitido en el hospital
    dentro de los próximos **30 días**, basado en información médica básica.
    """
)

# ==============================
# Datos de entrada del usuario
# ==============================
st.sidebar.header(" Datos del paciente")

age = st.sidebar.slider("Edad", 1, 100, 50)
gender = st.sidebar.selectbox("Género", ["Male", "Female"])
time_in_hospital = st.sidebar.slider("Días de hospitalización", 1, 14, 3)
num_lab_procedures = st.sidebar.slider("Número de procedimientos de laboratorio", 0, 150, 40)
num_medications = st.sidebar.slider("Número de medicamentos administrados", 0, 100, 10)
number_outpatient = st.sidebar.slider("Consultas ambulatorias", 0, 20, 2)
number_emergency = st.sidebar.slider("Número de visitas a urgencias", 0, 10, 1)
number_inpatient = st.sidebar.slider("Hospitalizaciones previas", 0, 10, 1)
A1Cresult = st.sidebar.selectbox("Resultado A1C", ["None", "Norm", ">7", ">8"])
insulin = st.sidebar.selectbox("Tratamiento con insulina", ["No", "Steady", "Up", "Down"])
change = st.sidebar.selectbox("¿Cambio de medicación?", ["No", "Ch"])

# Crear dataframe con los datos del usuario
df_input = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_medications": [num_medications],
    "number_outpatient": [number_outpatient],
    "number_emergency": [number_emergency],
    "number_inpatient": [number_inpatient],
    "A1Cresult": [A1Cresult],
    "insulin": [insulin],
    "change": [change]
})

st.subheader(" Datos ingresados")
st.dataframe(df_input)

# ==============================
# Predicción
# ==============================
st.subheader(" Resultado de la predicción")

if st.button("Predecir"):
    try:
        # 🔧 Asegurar que las columnas coincidan con el modelo entrenado
        try:
            expected_cols = model.feature_names_in_
        except AttributeError:
            expected_cols = (
                model.get_booster().feature_names
                if hasattr(model, "get_booster")
                else df_input.columns
            )

        # Agregar columnas faltantes con valor neutro
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        # Reordenar las columnas
        df_input = df_input.reindex(columns=expected_cols, fill_value=0)

        # 🔮 Realizar la predicción
        proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else 0
        threshold = 0.3  #  Ajuste del umbral para mejorar sensibilidad
        prediction = 1 if proba >= threshold else 0

        # Mostrar resultados con colores y mensajes claros
        if prediction == 1:
            st.success(f" El paciente **PODRÍA SER readmitido** en el hospital. (Probabilidad: {proba:.2%})")
        else:
            st.info(f" El paciente **probablemente NO será readmitido** en los próximos 30 días. (Probabilidad: {proba:.2%})")

        # Mostrar barra de probabilidad
        st.progress(int(proba * 100))
        st.caption(f"Probabilidad calculada: {proba:.2%}")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la predicción: {e}")

# ==============================
# Pie de página
# ==============================
st.markdown("---")

