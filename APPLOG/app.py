import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción de Compras - Regresión Logística", layout="wide")

# --- ENCABEZADO ---
st.markdown("""
<div style="background-color:#4B0082;padding:15px;border-radius:10px">
<h1 style="color:white;text-align:center;">Predicción de Compras de Usuarios</h1>
<h3 style="color:white;text-align:center;">Implementación de Regresión Logística</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Esta aplicación te permite:
- Subir tu **dataset (`UserData.csv`)**
- Subir tu **modelo entrenado (`modelo_regresion_logistica.pkl`)**
- Subir tu **escalador (`scaler.pkl`)**
- Realizar **predicciones en tiempo real** sobre nuevos usuarios
- Visualizar la **matriz de confusión y la curva ROC**
""")

# --- SUBIDA DE ARCHIVOS ---
uploaded_csv = st.file_uploader("Sube tu dataset (UserData.csv)", type=["csv"])
uploaded_model = st.file_uploader("Sube tu modelo entrenado (.pkl)", type=["pkl"])
uploaded_scaler = st.file_uploader("Sube tu scaler (.pkl)", type=["pkl"])

# --- MOSTRAR DATASET ---
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.markdown("### Vista previa de los datos")
    st.dataframe(df.head())

    if 'Purchased' in df.columns:
        # Mostrar distribución de clases
        st.markdown("###Distribución de Compras")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(x='Purchased', data=df, palette='coolwarm', ax=ax)
        ax.set_title("Distribución de la variable objetivo (Comprado)")
        st.pyplot(fig)

# --- CARGAR MODELO Y SCALER ---
if uploaded_model and uploaded_scaler:
    model = joblib.load(uploaded_model)
    scaler = joblib.load(uploaded_scaler)
    st.success("Modelo y scaler cargados correctamente.")

    # --- FORMULARIO DE PREDICCIÓN ---
    st.markdown("### Predicción de Nuevo Usuario")
    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad del usuario", min_value=18, max_value=70, value=30, step=1)
    with col2:
        salario = st.number_input("Salario estimado", min_value=10000, max_value=200000, value=50000, step=1000)

    if st.button("Predecir compra"):
        try:
            # Escalar los datos
            nuevo_dato = np.array([[edad, salario]])
            nuevo_dato_escalado = scaler.transform(nuevo_dato)

            # Predicción
            prediccion = model.predict(nuevo_dato_escalado)[0]
            prob = model.predict_proba(nuevo_dato_escalado)[0][1]

            resultado = "COMPRARÁ" if prediccion == 1 else "NO COMPRARÁ"

            st.markdown(f"### Resultado: **{resultado}**")
            st.write(f"**Probabilidad de compra:** {prob:.2%}")

            # Mostrar barra de probabilidad
            st.progress(int(prob * 100))
        except Exception as e:
            st.error(f"Error al predecir: {e}")

    # --- ANÁLISIS GRÁFICO (si hay dataset) ---
    if uploaded_csv:
        try:
            # Predecir todo el dataset
            X = df[['Age', 'EstimatedSalary']]
            X_scaled = scaler.transform(X)
            df['Predicted'] = model.predict(X_scaled)

            st.markdown("### Análisis de Resultados del Modelo")

            # Matriz de confusión
            from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

            cm = confusion_matrix(df['Purchased'], df['Predicted'])
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                        xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'], ax=ax)
            ax.set_title("Matriz de Confusión")
            st.pyplot(fig)

            # Curva ROC
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(df['Purchased'], y_pred_proba)
            auc = roc_auc_score(df['Purchased'], y_pred_proba)

            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
            ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
            ax.set_title('Curva ROC - Regresión Logística')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"No se pudieron generar los gráficos: {e}")

# --- MENSAJE FINAL ---
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
Regresión Logística (ACT XII)
</p>
""", unsafe_allow_html=True)
