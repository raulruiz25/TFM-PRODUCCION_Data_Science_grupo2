"""
Desarrollo del modelo predictivo (TFM - sección 8.2, capítulo 6 del índice)
Derivado de 06__Desarrollo_de_modelo_predictivo.ipynb

Construye las variables predictoras a partir de la salida del módulo 3
(dataset_consolidado_05.csv), entrena el modelo final (LightGBM, la
elección del TFM tras comparar contra XGBoost/Random Forest/LogisticRegression y
descartar LSTM y la arquitectura multimodal por bajo rendimiento), genera los
informes de evaluación e interpretabilidad (SHAP, contribución por familia de
variables, AUC por activo), serializa el modelo y calcula la predicción del
día para cada uno de los 6 activos con evidencia suficiente.

A petición expresa: se REENTRENA POR COMPLETO cada día (no se cachea el
modelo como en el módulo 2), para demostrar el pipeline de reentrenamiento
automático de principio a fin. El coste es bajo (~2.700 filas, modelos
ligeros), así que no compensa la complejidad de cachear.
"""

import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

LOCAL_DIR = Path("data_modelado")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_INPUT_PATH = ["TFM DATA SCIENCE", "03. data", "07. PROCESSED - Impacto Mercados"]
INPUT_FILENAME = "dataset_consolidado_05.csv"

DRIVE_OUTPUT_PATH = ["TFM DATA SCIENCE", "03. data", "08. PROCESSED - Modelado"]
DATASET_MODELADO_FILENAME = "dataset_modelado.csv"
MODELO_FILENAME = "modelo_evento_importante.pkl"
PREDICCIONES_FILENAME = "predicciones_hoy.csv"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

FECHA_FIN_TFM = pd.Timestamp("2026-06-30")
UMBRAL_DECISION = 0.5

FEATURE_COLS = [
    "ticker", "log_return", "volatility_20d", "volume_zscore_20d", "is_anomaly",
    "log_return_lag1", "log_return_lag2", "log_return_lag3", "log_return_lag4", "log_return_lag5",
    "volatility_lag1", "volatility_lag2", "volatility_lag3", "volatility_lag4", "volatility_lag5",
    "sentiment", "intensidad_max", "intensidad_media", "n_comunicaciones",
]
FEATURES_COMUNICACION = ["sentiment", "intensidad_max", "intensidad_media", "n_comunicaciones"]


# --------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y UTILIDADES DE DRIVE (mismo patrón que los módulos anteriores)
# --------------------------------------------------------------------------

def get_drive_service():
    key_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
    if not key_json:
        raise RuntimeError("No se encontró la variable de entorno GDRIVE_SERVICE_ACCOUNT_KEY.")
    key_info = json.loads(key_json)
    credentials = service_account.Credentials.from_service_account_info(key_info, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=credentials)


def resolver_carpeta_drive(drive_service, partes_ruta: list, crear_si_falta: bool = True) -> str:
    parent_id = None
    for i, nombre in enumerate(partes_ruta):
        query = f"name = '{nombre}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
        encontrados = resultado.get("files", [])
        if encontrados:
            parent_id = encontrados[0]["id"]
        elif crear_si_falta and i > 0:
            metadata = {"name": nombre, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
            carpeta = drive_service.files().create(body=metadata, fields="id").execute()
            parent_id = carpeta["id"]
        else:
            raise FileNotFoundError(f"No se encontró la carpeta '{nombre}' (ruta: {' / '.join(partes_ruta[:i + 1])}).")
    return parent_id


def buscar_archivo(drive_service, carpeta_id: str, nombre_archivo: str):
    query = f"name = '{nombre_archivo}' and '{carpeta_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
    encontrados = resultado.get("files", [])
    return encontrados[0]["id"] if encontrados else None


def descargar_archivo(drive_service, file_id: str, destino: Path):
    request = drive_service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    destino.write_bytes(buffer.getvalue())


def subir_o_actualizar_archivo(drive_service, ruta_local: Path, carpeta_id: str, nombre_archivo: str = None):
    nombre_archivo = nombre_archivo or ruta_local.name
    media = MediaFileUpload(str(ruta_local), resumable=True)
    existentes = drive_service.files().list(
        q=f"name = '{nombre_archivo}' and '{carpeta_id}' in parents and trashed = false",
        fields="files(id)"
    ).execute().get("files", [])
    if existentes:
        drive_service.files().update(fileId=existentes[0]["id"], media_body=media).execute()
    else:
        metadata = {"name": nombre_archivo, "parents": [carpeta_id]}
        drive_service.files().create(body=metadata, media_body=media, fields="id").execute()
    print(f"  Subido a Drive: {nombre_archivo}")


# --------------------------------------------------------------------------
# 3. CONSTRUCCIÓN DE VARIABLES PREDICTORAS (apartado 5)
# --------------------------------------------------------------------------

def construir_dataset_modelado(df_cap5: pd.DataFrame) -> pd.DataFrame:
    df_cap5 = df_cap5[df_cap5["date"] <= FECHA_FIN_TFM].reset_index(drop=True)
    df_cap5 = df_cap5.sort_values(["ticker", "date"]).reset_index(drop=True)

    for lag in range(1, 6):
        df_cap5[f"log_return_lag{lag}"] = df_cap5.groupby("ticker")["log_return"].shift(lag)
        df_cap5[f"volatility_lag{lag}"] = df_cap5.groupby("ticker")["volatility_20d"].shift(lag)

    df_cap5["sentiment"] = df_cap5["sentiment"].fillna(0.0)
    df_cap5["intensidad_max"] = df_cap5["intensidad_max"].fillna(0.0)
    df_cap5["intensidad_media"] = df_cap5["intensidad_media"].fillna(0.0)
    df_cap5["n_comunicaciones"] = df_cap5["n_comunicaciones"].fillna(0).astype(int)

    cols_lag = [c for c in df_cap5.columns if "_lag" in c]
    dataset_modelado = df_cap5.dropna(subset=cols_lag).reset_index(drop=True)
    return dataset_modelado


def preparar_X_y(df: pd.DataFrame, columnas_referencia=None):
    X = pd.get_dummies(df[FEATURE_COLS], columns=["ticker"], prefix="ticker")
    if columnas_referencia is not None:
        X = X.reindex(columns=columnas_referencia, fill_value=0)
    y = df["evento_importante"].astype(int)
    return X, y


# --------------------------------------------------------------------------
# 4. COMPARACIÓN DE MODELOS BASELINE Y VALIDACIÓN CRUZADA (apartado 6.1, informativo)
# --------------------------------------------------------------------------

def crear_modelos_baseline(y_train):
    peso = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
                                  eval_metric="logloss", scale_pos_weight=peso),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
                                    class_weight="balanced", verbose=-1),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
    }


def comparar_modelos_baseline(dataset_modelado):
    dataset_modelado = dataset_modelado.sort_values("date").reset_index(drop=True)
    fecha_corte = dataset_modelado["date"].quantile(0.8)
    train_df = dataset_modelado[dataset_modelado["date"] <= fecha_corte]
    test_df = dataset_modelado[dataset_modelado["date"] > fecha_corte]

    X_train, y_train = preparar_X_y(train_df)
    X_test, y_test = preparar_X_y(test_df, columnas_referencia=X_train.columns)

    filas = []
    for nombre, modelo in crear_modelos_baseline(y_train).items():
        modelo.fit(X_train, y_train)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        filas.append({
            "modelo": nombre,
            "accuracy": accuracy_score(y_test, modelo.predict(X_test)),
            "f1": f1_score(y_test, modelo.predict(X_test)),
            "auc": roc_auc_score(y_test, y_proba),
        })
    return pd.DataFrame(filas).sort_values("auc", ascending=False)


def validacion_cruzada_temporal(dataset_modelado, n_splits=5):
    X_full, y_full = preparar_X_y(dataset_modelado)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    resultados = {"XGBoost": [], "Random Forest": [], "LightGBM": [], "LogisticRegression": []}

    for train_idx, test_idx in tscv.split(X_full):
        y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]
        if y_te.nunique() < 2:
            continue
        X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
        for nombre, modelo in crear_modelos_baseline(y_tr).items():
            modelo.fit(X_tr, y_tr)
            resultados[nombre].append(roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1]))

    filas = [{"modelo": k, "auc_medio": np.mean(v) if v else np.nan,
              "auc_std": np.std(v) if v else np.nan, "n_splits_validos": len(v)}
             for k, v in resultados.items()]
    return pd.DataFrame(filas).sort_values("auc_medio", ascending=False)


# --------------------------------------------------------------------------
# 5. MODELO FINAL, EVALUACIÓN E INTERPRETABILIDAD (apartados 7-9)
# --------------------------------------------------------------------------

def entrenar_modelo_final(dataset_modelado):
    """LightGBM: la elección final del TFM (apartado 7.1) tras la comparación
    de modelos baseline — mejor equilibrio precisión/recall que Random Forest
    dado el desequilibrio de clases (evento_importante en solo el 9.5%)."""
    X_full, y_full = preparar_X_y(dataset_modelado)
    modelo_final = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   random_state=42, class_weight="balanced", verbose=-1)
    modelo_final.fit(X_full, y_full)
    return modelo_final, X_full, y_full


def evaluar_matriz_confusion(dataset_modelado):
    dataset_modelado = dataset_modelado.sort_values("date").reset_index(drop=True)
    fecha_corte = dataset_modelado["date"].quantile(0.8)
    train_df = dataset_modelado[dataset_modelado["date"] <= fecha_corte]
    test_df = dataset_modelado[dataset_modelado["date"] > fecha_corte]

    X_train, y_train = preparar_X_y(train_df)
    X_test, y_test = preparar_X_y(test_df, columnas_referencia=X_train.columns)

    modelo_eval = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  random_state=42, class_weight="balanced", verbose=-1)
    modelo_eval.fit(X_train, y_train)
    y_proba_test = modelo_eval.predict_proba(X_test)[:, 1]

    filas = []
    for umbral in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_proba_test >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        filas.append({
            "umbral": umbral, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        })
    df_matriz = pd.DataFrame(filas)
    auc_test = roc_auc_score(y_test, y_proba_test)
    return df_matriz, auc_test, modelo_eval, X_test, y_test


def calcular_shap_importancia(modelo_eval, X_test):
    explainer = shap.TreeExplainer(modelo_eval)
    shap_values = explainer.shap_values(X_test)
    shap_values_evento = shap_values[1] if isinstance(shap_values, list) \
        else shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    importancia = pd.DataFrame({
        "variable": X_test.columns,
        "importancia_media": np.abs(shap_values_evento).mean(axis=0),
    }).sort_values("importancia_media", ascending=False)

    peso_comunicacion = importancia[importancia["variable"].isin(FEATURES_COMUNICACION)]["importancia_media"].sum()
    peso_total = importancia["importancia_media"].sum()
    importancia["pct_del_total"] = importancia["importancia_media"] / peso_total * 100
    return importancia, peso_comunicacion / peso_total * 100


def contribucion_por_familia(dataset_modelado):
    dataset_modelado = dataset_modelado.sort_values("date").reset_index(drop=True)
    fecha_corte = dataset_modelado["date"].quantile(0.8)
    train_df = dataset_modelado[dataset_modelado["date"] <= fecha_corte]
    test_df = dataset_modelado[dataset_modelado["date"] > fecha_corte]

    X_train, y_train = preparar_X_y(train_df)
    X_test, y_test = preparar_X_y(test_df, columnas_referencia=X_train.columns)

    features_financieras = [c for c in X_train.columns if not any(c.startswith(v) for v in FEATURES_COMUNICACION)]
    features_comunicacion = FEATURES_COMUNICACION + [c for c in X_train.columns if c.startswith("ticker_")]

    filas = []
    for nombre, cols in [("Solo financieras", features_financieras),
                          ("Solo comunicación (+ ticker)", features_comunicacion),
                          ("Ambas", list(X_train.columns))]:
        modelo_temp = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      random_state=42, class_weight="balanced", verbose=-1)
        modelo_temp.fit(X_train[cols], y_train)
        proba_temp = modelo_temp.predict_proba(X_test[cols])[:, 1]
        filas.append({"familia": nombre, "n_features": len(cols), "auc": roc_auc_score(y_test, proba_temp)})
    return pd.DataFrame(filas)


def auc_por_activo(dataset_modelado):
    features_sin_ticker = [c for c in FEATURE_COLS if c != "ticker"]
    features_comunicacion_activo = ["intensidad_max", "intensidad_media", "n_comunicaciones"]
    features_financieras_activo = [c for c in features_sin_ticker if c not in FEATURES_COMUNICACION]

    filas = []
    for ticker in dataset_modelado["ticker"].unique():
        df_t = dataset_modelado[dataset_modelado["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        X_t = df_t[features_sin_ticker]
        y_t = df_t["evento_importante"].astype(int)

        fila = {"ticker": ticker, "n_eventos": int(y_t.sum())}
        for nombre, cols in [("financieras", features_financieras_activo),
                              ("comunicacion", features_comunicacion_activo),
                              ("ambas", features_sin_ticker)]:
            tscv = TimeSeriesSplit(n_splits=3)
            aucs = []
            for train_idx, test_idx in tscv.split(X_t):
                if y_t.iloc[test_idx].nunique() < 2:
                    continue
                modelo_t = LogisticRegression(class_weight="balanced", max_iter=1000)
                modelo_t.fit(X_t[cols].iloc[train_idx], y_t.iloc[train_idx])
                proba_t = modelo_t.predict_proba(X_t[cols].iloc[test_idx])[:, 1]
                aucs.append(roc_auc_score(y_t.iloc[test_idx], proba_t))
            fila[f"auc_{nombre}"] = np.mean(aucs) if aucs else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 6. SISTEMA DE PREDICCIÓN (apartado 10)
# --------------------------------------------------------------------------

def predecir_evento_importante(datos_nuevos: dict, modelo_info: dict) -> dict:
    modelo = modelo_info["modelo"]
    columnas_esperadas = modelo_info["columnas_esperadas"]
    umbral = modelo_info["umbral_decision"]

    df_input = pd.DataFrame([datos_nuevos])
    df_input_dummies = pd.get_dummies(df_input, columns=["ticker"], prefix="ticker")
    df_input_dummies = df_input_dummies.reindex(columns=columnas_esperadas, fill_value=0)

    probabilidad = modelo.predict_proba(df_input_dummies)[0, 1]
    return {
        "probabilidad": round(float(probabilidad), 4),
        "es_evento": bool(probabilidad >= umbral),
        "umbral_usado": umbral,
    }


# --------------------------------------------------------------------------
# 7. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------

def main():
    drive_service = get_drive_service()

    # 7.1 Cargar la salida del módulo 3
    carpeta_input_id = resolver_carpeta_drive(drive_service, DRIVE_INPUT_PATH)
    input_file_id = buscar_archivo(drive_service, carpeta_input_id, INPUT_FILENAME)
    if not input_file_id:
        raise FileNotFoundError(f"No se encontró '{INPUT_FILENAME}' en Drive (¿se ha ejecutado el módulo 3?).")
    ruta_input = LOCAL_DIR / INPUT_FILENAME
    descargar_archivo(drive_service, input_file_id, ruta_input)
    df_cap5 = pd.read_csv(ruta_input, parse_dates=["date"])
    print(f"Datos del módulo 3 cargados: {len(df_cap5)} filas, {df_cap5['ticker'].nunique()} activos.")

    # 7.2 Construir el dataset de modelado (apartado 5)
    dataset_modelado = construir_dataset_modelado(df_cap5)
    print(f"Dataset de modelado: {len(dataset_modelado)} filas "
          f"({dataset_modelado['evento_importante'].mean() * 100:.1f}% eventos).")

    carpeta_output_id = resolver_carpeta_drive(drive_service, DRIVE_OUTPUT_PATH)
    ruta_modelado = LOCAL_DIR / DATASET_MODELADO_FILENAME
    dataset_modelado.to_csv(ruta_modelado, index=False)
    subir_o_actualizar_archivo(drive_service, ruta_modelado, carpeta_output_id, DATASET_MODELADO_FILENAME)

    # 7.3 Comparación de modelos y validación cruzada (informativo, para monitorización)
    print("Comparando modelos baseline (6.1)...")
    comparacion = comparar_modelos_baseline(dataset_modelado)
    print("Validación cruzada temporal (5 splits)...")
    cv_resultados = validacion_cruzada_temporal(dataset_modelado)

    # 7.4 Modelo final: reentrenamiento completo (Random Forest, elección del TFM)
    print("Reentrenando el modelo final (Random Forest) sobre el histórico completo...")
    modelo_final, X_full, y_full = entrenar_modelo_final(dataset_modelado)

    # 7.5 Evaluación (matriz de confusión, AUC en test real)
    print("Evaluando sobre partición temporal de test...")
    df_matriz, auc_test, modelo_eval, X_test, y_test = evaluar_matriz_confusion(dataset_modelado)
    print(f"  AUC sobre test real: {auc_test:.3f}")

    # 7.6 Interpretabilidad: SHAP + contribución por familia + AUC por activo
    print("Calculando importancia SHAP...")
    importancia_shap, pct_comunicacion = calcular_shap_importancia(modelo_eval, X_test)
    print(f"  Peso de las variables de comunicación: {pct_comunicacion:.1f}%")

    print("Calculando contribución por familia de variables...")
    contribucion = contribucion_por_familia(dataset_modelado)

    print("Calculando AUC por activo (financieras vs. comunicación vs. ambas)...")
    auc_activo = auc_por_activo(dataset_modelado)

    # 7.7 Guardar informes en Drive
    informes = {
        "informe_comparacion_modelos.csv": comparacion,
        "informe_cv_temporal.csv": cv_resultados,
        "informe_matriz_confusion_umbrales.csv": df_matriz,
        "informe_shap_importancia.csv": importancia_shap,
        "informe_contribucion_familias.csv": contribucion,
        "informe_auc_por_activo.csv": auc_activo,
    }
    for nombre, df_informe in informes.items():
        ruta = LOCAL_DIR / nombre
        df_informe.to_csv(ruta, index=False)
        subir_o_actualizar_archivo(drive_service, ruta, carpeta_output_id, nombre)

    # 7.8 Serializar el modelo final
    modelo_info = {
        "modelo": modelo_final,
        "columnas_esperadas": list(X_full.columns),
        "umbral_decision": UMBRAL_DECISION,
        "feature_cols_originales": FEATURE_COLS,
    }
    ruta_modelo = LOCAL_DIR / MODELO_FILENAME
    joblib.dump(modelo_info, ruta_modelo)
    subir_o_actualizar_archivo(drive_service, ruta_modelo, carpeta_output_id, MODELO_FILENAME)

    # 7.9 Predicción del día para cada activo (última fila disponible de cada uno)
    print("Calculando la predicción de hoy para cada activo...")
    predicciones = []
    for ticker in dataset_modelado["ticker"].unique():
        ultima_fila = dataset_modelado[dataset_modelado["ticker"] == ticker].sort_values("date").iloc[-1]
        datos_ejemplo = {col: ultima_fila[col] for col in FEATURE_COLS}
        resultado = predecir_evento_importante(datos_ejemplo, modelo_info)
        predicciones.append({
            "ticker": ticker, "fecha": ultima_fila["date"],
            "probabilidad": resultado["probabilidad"], "es_evento": resultado["es_evento"],
        })
        print(f"  [{ticker}] {ultima_fila['date'].date()}: probabilidad={resultado['probabilidad']}, "
              f"es_evento={resultado['es_evento']}")

    df_predicciones = pd.DataFrame(predicciones)
    ruta_predicciones = LOCAL_DIR / PREDICCIONES_FILENAME
    df_predicciones.to_csv(ruta_predicciones, index=False)
    subir_o_actualizar_archivo(drive_service, ruta_predicciones, carpeta_output_id, PREDICCIONES_FILENAME)

    print("Desarrollo del modelo predictivo completado.")


if __name__ == "__main__":
    main()
