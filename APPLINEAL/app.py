import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción de Salarios", layout="wide")

# Diseño del encabezado
st.markdown("""
<div style="background-color:#8A2BE2;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">Predicción de Salarios según Años de Experiencia</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Esta aplicación permite:
- Subir tu **notebook (.ipynb)**  
- Subir tu **dataset (.csv)**  
- Subir tu **modelo entrenado (.pkl)** para hacer predicciones.
""")

# Subida de archivos
uploaded_notebook = st.file_uploader("Sube tu Notebook (.ipynb)", type=["ipynb"])
uploaded_csv = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])
uploaded_model = st.file_uploader("Sube tu modelo entrenado (.pkl)", type=["pkl"])

# Mostrar tabla y gráfica del CSV
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.markdown("### Vista previa de los datos")
    st.dataframe(df)

    try:
        st.markdown("### Gráfica: YearsExperience vs Salary")
        fig, ax = plt.subplots(figsize=(5,3)) 
        ax.scatter(df['YearsExperience'], df['Salary'], color="blue", alpha=0.7, label="Datos reales")
        ax.set_xlabel("YearsExperience")
        ax.set_ylabel("Salary")
        ax.set_title("Relación entre experiencia y salario")

        # Si hay modelo cargado, dibujar la línea de regresión
        if uploaded_model:
            model = joblib.load(uploaded_model)
            x_sorted = df[['YearsExperience']].sort_values(by='YearsExperience')
            y_pred = model.predict(x_sorted)
            ax.plot(x_sorted, y_pred, color="red", linewidth=2, label="Línea de regresión")

        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"No se pudo graficar: {e}")

# Cargar modelo y predecir
if uploaded_model:
    model = joblib.load(uploaded_model)
    st.success("Modelo cargado correctamente.")

    st.markdown("### Predicción de Salario")
    experiencia = st.number_input(
        "Años de experiencia",
        min_value=int(df['YearsExperience'].min()) if uploaded_csv else 0,
        max_value=int(df['YearsExperience'].max()) if uploaded_csv else 40,
        step=1
    )

    if st.button("Predecir salario"):
        try:
            pred = model.predict([[experiencia]])[0]
            st.success(f"Salario predicho: ${pred:,.2f}")

            # Mostrar punto en la gráfica
            fig, ax = plt.subplots(figsize=(5,3))
            ax.scatter(df['YearsExperience'], df['Salary'], color="blue", alpha=0.7, label="Datos reales")

            # Línea del modelo
            x_sorted = df[['YearsExperience']].sort_values(by='YearsExperience')
            y_pred = model.predict(x_sorted)
            ax.plot(x_sorted, y_pred, color="red", linewidth=2, label="Línea de regresión")

            # Punto de la predicción
            ax.scatter(experiencia, pred, color="green", s=100, label="Predicción actual")

            ax.set_xlabel("YearsExperience")
            ax.set_ylabel("Salary")
            ax.set_title("Predicción según experiencia")
            ax.legend()
            st.pyplot(fig)

            # Tabla con predicción
            nueva_fila = pd.DataFrame({
                "YearsExperience": [experiencia],
                "Salary": [pred]
            })
            if uploaded_csv:
                df_pred = pd.concat([df, nueva_fila], ignore_index=True)
            else:
                df_pred = nueva_fila
            st.markdown("### Tabla con salarios incluyendo predicción")
            st.dataframe(df_pred)
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# Mostrar notebook subido
if uploaded_notebook:
    st.markdown("### Notebook subido correctamente")
    st.info("El archivo se ha cargado (solo se muestra como evidencia, no se ejecuta).")
    st.download_button("Descargar Notebook", uploaded_notebook, file_name="Lab3_RegresionLineal.ipynb")
