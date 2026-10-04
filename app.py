import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ui import (
    aplicar_estilos,
    panel,
    render_hero,
    render_metricas,
    render_parametros,
    render_precio,
    render_resultado,
    render_status,
    texto_ayuda,
    titulo_seccion,
)


BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "content" / "Used_Car_Prices.csv"
RUTA_ARTEFACTOS = BASE_DIR / "content" / "artefactos_modelo.joblib"

COLUMNAS_ELIMINAR = [
    "RecordID",
    "FullName",
    "Phone",
    "ZodiacSign",
    "FavoriteColor",
    "Hobby",
    "PriceWithTax",
    "Mileage_km",
    "City",
]
COLUMNAS_NUMERICAS = [
    "ModelYear",
    "Mileage_miles",
    "EngineSize",
    "Horsepower",
    "Doors",
    "PreviousOwners",
]
COLUMNAS_CATEGORICAS = ["Brand", "FuelType", "Transmission"]


def cap_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = max(0, q1 - 1.5 * iqr)
    limite_superior = q3 + 1.5 * iqr
    return np.clip(series, limite_inferior, limite_superior), limite_inferior, limite_superior


def calcular_mape(y_real, y_predicho):
    y_real = np.array(y_real)
    y_predicho = np.array(y_predicho)
    mascara = y_real != 0
    return np.mean(np.abs((y_real[mascara] - y_predicho[mascara]) / y_real[mascara])) * 100


def limpiar_datos(df_crudo):
    df = df_crudo.drop(columns=COLUMNAS_ELIMINAR, errors="ignore").copy()

    limites = {}
    for col in ["Mileage_miles", "Horsepower", "Price"]:
        df[col], lim_inf, lim_sup = cap_outliers(df[col])
        limites[col] = {"inferior": lim_inf, "superior": lim_sup}

    medianas = {}
    for col in ["Mileage_miles", "Horsepower", "EngineSize"]:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        medianas[col] = mediana

    moda_brand = df["Brand"].mode()[0]
    df["Brand"] = df["Brand"].fillna(moda_brand)

    return df, {
        "limites": limites,
        "medianas": medianas,
        "moda_brand": moda_brand,
    }


def codificar_categoricas(df):
    df_codificado = pd.get_dummies(df, columns=COLUMNAS_CATEGORICAS, drop_first=True)
    for col in df_codificado.columns:
        if df_codificado[col].dtype == "bool":
            df_codificado[col] = df_codificado[col].astype(int)
    return df_codificado


def entrenar_y_evaluar(df_limpio):
    df = codificar_categoricas(df_limpio)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLUMNAS_NUMERICAS] = scaler.fit_transform(X_train[COLUMNAS_NUMERICAS])
    X_test[COLUMNAS_NUMERICAS] = scaler.transform(X_test[COLUMNAS_NUMERICAS])

    modelos_params = {
        "Regresion_Lineal": (LinearRegression(), {}),
        "Arbol_Decision": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [5, 10, 15]},
        ),
        "SVR": (SVR(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
        "Red_Neuronal_MLP": (
            MLPRegressor(max_iter=1000, random_state=42),
            {"hidden_layer_sizes": [(50,), (100,)]},
        ),
        "Bagging_RandomForest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [50, 100], "max_depth": [5, 10]},
        ),
        "Boosting_Gradient": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        ),
    }

    mejores_modelos = {}
    mejores_params = {}

    for nombre, (modelo, params) in modelos_params.items():
        grid = GridSearchCV(
            modelo,
            params,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        mejores_modelos[nombre] = grid.best_estimator_
        mejores_params[nombre] = grid.best_params_

    modelo_voting = VotingRegressor(
        estimators=[
            ("rf", mejores_modelos["Bagging_RandomForest"]),
            ("gb", mejores_modelos["Boosting_Gradient"]),
            ("lr", mejores_modelos["Regresion_Lineal"]),
        ]
    )
    modelo_voting.fit(X_train, y_train)
    mejores_modelos["Ensamble_Votacion"] = modelo_voting
    mejores_params["Ensamble_Votacion"] = "Combinación de RF + Gradient Boosting + Regresión Lineal"

    resultados = []
    for nombre, modelo in mejores_modelos.items():
        y_pred = modelo.predict(X_test)
        resultados.append(
            {
                "Modelo": nombre,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
                "MAPE (%)": calcular_mape(y_test, y_pred),
            }
        )

    df_resultados = pd.DataFrame(resultados).sort_values(by="MAE").reset_index(drop=True)
    mejor_nombre = df_resultados.iloc[0]["Modelo"]
    mejor_modelo = mejores_modelos[mejor_nombre]

    return {
        "modelo": mejor_modelo,
        "nombre_modelo": mejor_nombre,
        "scaler": scaler,
        "columnas": list(X_train.columns),
        "resultados": df_resultados,
        "mejores_params": mejores_params,
    }


def preparar_para_prediccion(df_entrada, artefactos):
    df = df_entrada.copy()

    for col in COLUMNAS_ELIMINAR + ["Price"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    info = artefactos["info_limpieza"]

    for col in ["Mileage_miles", "Horsepower"]:
        lim = info["limites"][col]
        df[col] = np.clip(df[col], lim["inferior"], lim["superior"])

    for col, mediana in info["medianas"].items():
        df[col] = df[col].fillna(mediana)

    df["Brand"] = df["Brand"].fillna(info["moda_brand"])

    df = codificar_categoricas(df)

    for col in artefactos["columnas"]:
        if col not in df.columns:
            df[col] = 0

    df = df[artefactos["columnas"]]
    df[COLUMNAS_NUMERICAS] = artefactos["scaler"].transform(df[COLUMNAS_NUMERICAS])
    return df


@st.cache_resource(show_spinner=False)
def cargar_o_entrenar_modelo():
    if RUTA_ARTEFACTOS.exists():
        return joblib.load(RUTA_ARTEFACTOS)

    df_crudo = pd.read_csv(RUTA_DATOS)
    df_limpio, info_limpieza = limpiar_datos(df_crudo)
    resultado = entrenar_y_evaluar(df_limpio)

    artefactos = {
        "modelo": resultado["modelo"],
        "nombre_modelo": resultado["nombre_modelo"],
        "scaler": resultado["scaler"],
        "columnas": resultado["columnas"],
        "resultados": resultado["resultados"],
        "mejores_params": resultado["mejores_params"],
        "info_limpieza": info_limpieza,
        "opciones": {
            "Brand": sorted(df_limpio["Brand"].dropna().unique().tolist()),
            "FuelType": sorted(df_limpio["FuelType"].dropna().unique().tolist()),
            "Transmission": sorted(df_limpio["Transmission"].dropna().unique().tolist()),
        },
    }

    joblib.dump(artefactos, RUTA_ARTEFACTOS)
    return artefactos


def formatear_tabla(df, redondeos=None):
    vista = df.copy()
    if redondeos:
        for col, decimales in redondeos.items():
            if col in vista.columns:
                vista[col] = vista[col].astype(float).round(decimales)
    return vista


def mostrar_evaluacion(artefactos):
    titulo_seccion("sliders", "4.2 Ajuste de hiperparámetros (Cross Validation)")
    panel(
        """
        <p class="hint" style="margin:0;">
            Se usó el <strong>70%</strong> de los datos para entrenar.
            La métrica de optimización en GridSearchCV fue
            <code>neg_mean_absolute_error</code> (MAE), porque el error se interpreta
            directamente en dinero (unidades del precio).
        </p>
        """
    )

    with st.expander("Ver mejores hiperparámetros por modelo"):
        render_parametros(artefactos["mejores_params"])

    titulo_seccion("clipboard", "4.3 Medida de calidad del modelo")

    df_resultados = artefactos["resultados"].copy()
    mejor = df_resultados.iloc[0]

    render_metricas(artefactos["nombre_modelo"], mejor["MAE"], mejor["R2"])
    panel(
        """
        <p class="hint" style="margin:0;">
            Evaluación sobre el <strong>30%</strong> de prueba.
            El modelo ganador se elige por el menor MAE.
        </p>
        """
    )

    vista = formatear_tabla(
        df_resultados,
        {"MAE": 2, "RMSE": 2, "R2": 4, "MAPE (%)": 2},
    )
    st.dataframe(vista, width="stretch", hide_index=True)

    render_resultado(
        f'Modelo seleccionado: <strong>{artefactos["nombre_modelo"]}</strong>'
        f'&nbsp;·&nbsp; MAE = {mejor["MAE"]:.2f}'
    )


def pestana_prediccion_manual(artefactos):
    titulo_seccion("car", "Ingreso manual de un vehículo")
    texto_ayuda("Completa las características del vehículo para estimar su precio de venta.")

    opciones = artefactos["opciones"]

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marca (Brand)", opciones["Brand"])
        model_year = st.number_input(
            "Año modelo (ModelYear)", min_value=1990, max_value=2026, value=2018
        )
        mileage = st.number_input(
            "Kilometraje en millas (Mileage_miles)",
            min_value=0.0,
            value=50000.0,
            step=100.0,
        )
        fuel = st.selectbox("Combustible (FuelType)", opciones["FuelType"])
        transmission = st.selectbox("Transmisión (Transmission)", opciones["Transmission"])

    with col2:
        engine = st.number_input(
            "Cilindraje (EngineSize)", min_value=0.6, max_value=8.0, value=2.0, step=0.1
        )
        horsepower = st.number_input(
            "Potencia (Horsepower)", min_value=40.0, max_value=700.0, value=150.0, step=1.0
        )
        doors = st.selectbox("Puertas (Doors)", [2, 3, 4, 5], index=2)
        owners = st.number_input(
            "Dueños previos (PreviousOwners)", min_value=0, max_value=10, value=1
        )

    if st.button("Predecir precio", type="primary"):
        df_entrada = pd.DataFrame(
            [
                {
                    "Brand": brand,
                    "ModelYear": model_year,
                    "Mileage_miles": mileage,
                    "FuelType": fuel,
                    "Transmission": transmission,
                    "EngineSize": engine,
                    "Horsepower": horsepower,
                    "Doors": doors,
                    "PreviousOwners": owners,
                }
            ]
        )

        try:
            df_procesado = preparar_para_prediccion(df_entrada, artefactos)
            prediccion = artefactos["modelo"].predict(df_procesado)[0]

            titulo_seccion("table", "Datos enviados al modelo")
            st.dataframe(
                formatear_tabla(df_procesado, {c: 4 for c in df_procesado.columns}),
                width="stretch",
                hide_index=True,
            )
            render_precio(prediccion)
        except Exception as e:
            st.error(f"No se pudo realizar la predicción: {e}")


def pestana_prediccion_archivo(artefactos):
    titulo_seccion("file", "Predicción por lotes (CSV)")
    texto_ayuda(
        "Sube un archivo CSV con las mismas columnas del dataset original "
        "(o al menos las variables usadas por el modelo). "
        "No se necesita la columna <code>Price</code>."
    )

    archivo = st.file_uploader("Archivo CSV", type=["csv"])

    if archivo is None:
        return

    try:
        df_excel = pd.read_csv(archivo)
        titulo_seccion("eye", "Vista previa de los datos cargados")
        st.dataframe(df_excel.head(), width="stretch", hide_index=True)

        if st.button("Predecir por lotes", type="primary"):
            with st.spinner("Procesando y prediciendo..."):
                df_procesado = preparar_para_prediccion(df_excel, artefactos)
                predicciones = artefactos["modelo"].predict(df_procesado)

                df_salida = df_excel.copy()
                df_salida["Precio_Predicho"] = np.round(predicciones, 2)

                render_resultado("Predicciones completadas correctamente.")
                st.dataframe(df_salida, width="stretch", hide_index=True)

                csv_bytes = df_salida.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar resultados CSV",
                    data=csv_bytes,
                    file_name="predicciones_precios.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")


def main():
    st.set_page_config(
        page_title="Predicción de Precios de Vehículos Usados",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    aplicar_estilos()
    render_hero()

    if not RUTA_DATOS.exists():
        st.error(f"No se encontró el dataset en: `{RUTA_DATOS}`")
        st.stop()

    with st.spinner("Cargando o entrenando modelos (solo la primera vez tarda unos minutos)..."):
        try:
            artefactos = cargar_o_entrenar_modelo()
        except Exception as e:
            st.error(f"Error al preparar el modelo: {e}")
            st.stop()

    render_status(artefactos["nombre_modelo"])

    tab1, tab2, tab3 = st.tabs(
        [
            "Evaluación de modelos",
            "Predicción manual",
            "Predicción por archivo",
        ]
    )

    with tab1:
        mostrar_evaluacion(artefactos)

    with tab2:
        pestana_prediccion_manual(artefactos)

    with tab3:
        pestana_prediccion_archivo(artefactos)


if __name__ == "__main__":
    main()
