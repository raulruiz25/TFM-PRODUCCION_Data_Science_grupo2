"""
Análisis del impacto en los mercados (TFM - sección 8.2, apartado 5 del índice)
Derivado de 05_analisis_impacto_mercados_financieros.ipynb

Recalcula, con los datos financieros y semánticos más recientes disponibles en
Drive: estudio de eventos (5.1), impacto pre/post en precio y volumen (5.2/5.3),
correlación sentimiento-retorno (5.4), y detección de anomalías con
autoencoders + cruce con comunicaciones (5.5). Además genera el dataset final
que alimenta el capítulo 6 del TFM.

A diferencia de los módulos 1 y 2, este NO es incremental: se recalcula por
completo en cada ejecución, porque los modelos estadísticos (ventanas de
estimación del event study, autoencoders) dependen de toda la serie histórica,
no solo de las filas nuevas. El coste de recalcular todo es bajo (estadística
+ autoencoders pequeños, sin modelos de lenguaje), así que no compensa la
complejidad de hacerlo incremental.

Se ejecuta automáticamente todos los días (encadenado tras la ingesta
financiera), para demostrar que el pipeline completo funcionaría de forma
automática si también se dispusiera de acceso diario a comunicaciones nuevas
-- hoy el corpus de texto es fijo, así que solo cambian ligeramente los
resultados financieros día a día, pero el mecanismo queda demostrado.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

LOCAL_DIR = Path("data_impacto")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_FINANCIERO_PATH = ["Grupo 2_Predicción_Movimientos_Mercado_&_Comunicaciones_Públicas", "03. data", "01. RAW - Datos Financieros"]
DRIVE_SEMANTICO_PATH = ["Grupo 2_Predicción_Movimientos_Mercado_&_Comunicaciones_Públicas", "03. data", "05. PROCESSED - Analisis Semantico"]
DRIVE_OUTPUT_PATH = ["Grupo 2_Predicción_Movimientos_Mercado_&_Comunicaciones_Públicas", "03. data", "07. PROCESSED - Impacto Mercados"]

SEMANTICO_FILENAME = "dataset_semantico.csv"
OUTPUT_FILENAME = "dataset_consolidado_05.csv"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

ASSETS = {
    'GSPC': 'GSPC.csv', 'IXIC': 'IXIC.csv', 'TSLA': 'TSLA.csv', 'VIX': 'VIX.csv',
    'XLV': 'XLV.csv', 'XLF': 'XLF.csv', 'XLE': 'XLE.csv', 'ITA': 'ITA.csv',
    'BTC-USD': 'BTC-USD.csv', 'ETH-USD': 'ETH-USD.csv', 'GC=F': 'GC=F.csv',
}
ASSETS_SIN_VOLUMEN = {'VIX'}

WINDOW_SIZE = 20
TRAIN_SPLIT = 0.8
ANOMALY_PERCENTILE = 95
FEATURE_COLS = ['log_return', 'high_low_range', 'volatility_20d', 'volume_zscore_20d']

TICKER_CATEGORIAS = {
    'GSPC': None, 'IXIC': None, 'VIX': None, 'BTC-USD': None, 'ETH-USD': None,
    'XLE': ['ENERGIA_MATERIAS', 'GEOPOLITICA'],
    'GC=F': ['ENERGIA_MATERIAS', 'MACRO_FISCAL'],
    'TSLA': ['EMPRESAS', 'ARANCELES_COMERCIO'],
    'XLF': ['POLITICA_MONETARIA', 'MACRO_FISCAL', 'REGULACION'],
    'XLV': ['REGULACION', 'EMPRESAS'],
    'ITA': ['GEOPOLITICA'],
}

UMBRAL_EVENTO_DESV = 2.0
UMBRAL_INTENSIDAD = 0.5

# Decisión metodológica fija del TFM (capítulo 6): no se recalcula cada día,
# ver cabecera del fichero para la justificación.
ACTIVOS_CON_EVIDENCIA = ['IXIC', 'XLE', 'TSLA', 'GSPC', 'ETH-USD', 'BTC-USD']


# --------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y UTILIDADES DE DRIVE
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


def subir_archivo(drive_service, ruta_local: Path, carpeta_id: str, nombre_archivo: str = None):
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
# 3. CARGA Y FEATURES (funciones comunes)
# --------------------------------------------------------------------------

def load_and_clean(csv_path, ticker):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    vol_col = 'VOLUME_USD' if 'VOLUME_USD' in df.columns else 'VOLUME_UNITS'
    df = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', vol_col]].rename(columns={vol_col: 'VOLUME'})
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['ticker'] = ticker
    return df


def compute_features(df):
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_20d'] = df['log_return'].rolling(window=20).std()

    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    with np.errstate(invalid='ignore', divide='ignore'):
        zscore = (df['volume'] - vol_mean) / vol_std
    df['volume_zscore_20d'] = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if df['ticker'].iloc[0] in ASSETS_SIN_VOLUMEN:
        df['volume_zscore_20d'] = 0.0

    return df.dropna(subset=['log_return', 'high_low_range', 'volatility_20d']).reset_index(drop=True)


def filtrar_comunicaciones_relevantes(ticker, df_semantico):
    base = df_semantico[df_semantico['MARKET_IMPACT'] == 1]
    categorias = TICKER_CATEGORIAS.get(ticker)
    if categorias is None:
        return base
    patron = '|'.join(categorias)
    return base[base['IMPACT_CATEGORIES'].fillna('').str.contains(patron)]


# --------------------------------------------------------------------------
# 4. ESTUDIO DE EVENTOS (5.1) Y PRE/POST (5.2-5.3)
# --------------------------------------------------------------------------

def event_study(data, ticker, event_dates, window=(-5, 5), estimation_window=(-250, -30),
                 benchmark_ticker='GSPC', min_estimation_obs=30):
    df = data[ticker].reset_index(drop=True)
    dates = df['date'].values
    log_returns = df['log_return'].values

    use_benchmark = benchmark_ticker in data and benchmark_ticker != ticker
    if use_benchmark:
        bdf = data[benchmark_ticker][['date', 'log_return']].rename(columns={'log_return': 'log_return_mkt'}).sort_values('date')
        merged_full = pd.merge_asof(df[['date', 'log_return']].sort_values('date'), bdf,
                                     on='date', direction='nearest', tolerance=pd.Timedelta('3D'))
    else:
        merged_full = df[['date', 'log_return']].copy()
        merged_full['log_return_mkt'] = np.nan

    event_records = []
    for event_date in event_dates:
        event_date = pd.Timestamp(event_date).normalize()
        idx_evt = np.searchsorted(dates, np.datetime64(event_date))
        est_start, est_end = idx_evt + estimation_window[0], idx_evt + estimation_window[1]
        win_start, win_end = idx_evt + window[0], idx_evt + window[1] + 1

        if idx_evt <= 0 or idx_evt >= len(dates) or est_start < 0 or win_end > len(dates):
            continue

        est_slice = merged_full.iloc[est_start:est_end]
        n_mkt_obs = est_slice['log_return_mkt'].notna().sum()

        if use_benchmark and n_mkt_obs >= min_estimation_obs:
            valid = est_slice.dropna(subset=['log_return_mkt'])
            beta, alpha = np.polyfit(valid['log_return_mkt'], valid['log_return'], 1)
            modelo = 'mercado'
        else:
            alpha, beta = est_slice['log_return'].mean(), 0.0
            modelo = 'media'

        window_returns = log_returns[win_start:win_end]
        window_mkt = merged_full.iloc[win_start:win_end]['log_return_mkt'].values

        if modelo == 'mercado':
            expected = alpha + beta * window_mkt
            expected = np.where(np.isnan(window_mkt), alpha, expected)
        else:
            expected = np.full(len(window_returns), alpha)

        AR = window_returns - expected
        CAR = np.cumsum(AR)
        event_records.append({'event_date': event_date, 'CAR_final': CAR[-1], 'modelo': modelo})

    return pd.DataFrame(event_records)


def run_event_study_all(data, df_semantico):
    summary_rows = []
    for ticker in data.keys():
        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico)
        event_dates_ticker = comms_ticker['TIMESTAMP'].dt.normalize().drop_duplicates().sort_values().tolist()

        events_df = event_study(data, ticker, event_dates_ticker)
        if events_df.empty:
            continue

        t_stat, p_value = stats.ttest_1samp(events_df['CAR_final'], 0)
        summary_rows.append({
            'ticker': ticker, 'n_fechas_evento': len(event_dates_ticker),
            'n_eventos_analizados': len(events_df),
            'n_eventos_modelo_media': int((events_df['modelo'] == 'media').sum()),
            'CAR_medio_final': events_df['CAR_final'].mean(),
            't_stat': t_stat, 'p_value': p_value, 'significativo_5pct': p_value < 0.05,
        })
    return pd.DataFrame(summary_rows).sort_values('p_value')


def pre_post_comparison(data, ticker, event_dates, pre_window=5, post_window=5):
    df = data[ticker].reset_index(drop=True)
    dates = df['date'].values
    pre_vol, post_vol, pre_volu, post_volu = [], [], [], []

    for event_date in event_dates:
        event_date = pd.Timestamp(event_date).normalize()
        idx_evt = np.searchsorted(dates, np.datetime64(event_date))
        if idx_evt - pre_window < 0 or idx_evt + post_window >= len(dates):
            continue
        pre_vol.append(df['volatility_20d'].iloc[idx_evt - pre_window:idx_evt].mean())
        post_vol.append(df['volatility_20d'].iloc[idx_evt:idx_evt + post_window].mean())
        pre_volu.append(df['volume_zscore_20d'].iloc[idx_evt - pre_window:idx_evt].mean())
        post_volu.append(df['volume_zscore_20d'].iloc[idx_evt:idx_evt + post_window].mean())

    if len(pre_vol) < 2:
        return {'ticker': ticker, 'n_eventos': len(pre_vol)}

    t_vol, p_vol = stats.ttest_rel(post_vol, pre_vol)
    t_volu, p_volu = stats.ttest_rel(post_volu, pre_volu)
    return {
        'ticker': ticker, 'n_eventos': len(pre_vol),
        'volatilidad_pre': np.mean(pre_vol), 'volatilidad_post': np.mean(post_vol),
        'volatilidad_p_value': p_vol, 'volatilidad_significativo_5pct': p_vol < 0.05,
        'volumen_pre': np.mean(pre_volu), 'volumen_post': np.mean(post_volu),
        'volumen_p_value': p_volu, 'volumen_significativo_5pct': p_volu < 0.05,
    }


def run_pre_post_all(data, df_semantico):
    filas = []
    for ticker in data.keys():
        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico)
        event_dates_ticker = comms_ticker['TIMESTAMP'].dt.normalize().drop_duplicates().sort_values().tolist()
        filas.append(pre_post_comparison(data, ticker, event_dates_ticker))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 5. CORRELACIÓN SEMÁNTICA (5.4)
# --------------------------------------------------------------------------

def semantic_market_correlation_1d(data, ticker, comunicaciones_df, sentiment_col='sentimiento_continuo', date_col='date'):
    df_mkt = data[ticker][['date', 'log_return']].copy()
    comm = comunicaciones_df.copy()
    comm[date_col] = pd.to_datetime(comm[date_col]).dt.normalize()
    comm_daily = comm.groupby(date_col)[sentiment_col].mean().reset_index()

    merged = df_mkt.merge(comm_daily, left_on='date', right_on=date_col, how='inner')
    if merged.empty:
        return None
    merged['log_return_lag1'] = merged['log_return'].shift(-1)
    valid = merged[[sentiment_col, 'log_return_lag1']].dropna()

    if len(valid) > 2:
        r, p_value = pearsonr(valid[sentiment_col], valid['log_return_lag1'])
    else:
        r, p_value = np.nan, np.nan
    return {'ticker': ticker, 'correlacion': r, 'p_value': p_value, 'n_obs': len(valid)}


def run_semantic_corr_all_1d(data, df_semantico):
    filas = []
    for ticker in data.keys():
        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico).rename(columns={'TIMESTAMP': 'date'})
        resultado = semantic_market_correlation_1d(data, ticker, comms_ticker)
        if resultado is not None:
            filas.append(resultado)
    summary = pd.DataFrame(filas).sort_values('p_value')
    if not summary.empty:
        rechazado, p_corregido, _, _ = multipletests(summary['p_value'], alpha=0.05, method='fdr_bh')
        summary['p_value_corregido'] = p_corregido
        summary['significativo_tras_correccion'] = rechazado
    return summary


# --------------------------------------------------------------------------
# 6. AUTOENCODERS (5.5.1-5.5.2)
# --------------------------------------------------------------------------

def create_windows(data_arr, window_size):
    return np.array([data_arr[i:i + window_size] for i in range(len(data_arr) - window_size + 1)])


def prepare_windows(df, feature_cols=FEATURE_COLS, window_size=WINDOW_SIZE, train_split=TRAIN_SPLIT):
    values = df[feature_cols].values
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)
    X_windows = create_windows(values_scaled, window_size)
    window_dates = df['date'].values[window_size - 1:]
    split_idx = int(len(X_windows) * train_split)
    return {
        'X_train': X_windows[:split_idx], 'X_test': X_windows[split_idx:],
        'dates_train': window_dates[:split_idx], 'dates_test': window_dates[split_idx:],
    }


def build_autoencoder(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'), layers.Dense(8, activation='relu'),
        layers.Dense(32, activation='relu'), layers.Dense(input_dim, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_autoencoder(windows_dict, epochs=100, batch_size=16):
    X_train, X_test = windows_dict['X_train'], windows_dict['X_test']
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    model = build_autoencoder(X_train_flat.shape[1])
    model.fit(X_train_flat, X_train_flat, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
    windows_dict['X_train_flat'] = X_train_flat
    windows_dict['X_test_flat'] = X_test_flat
    return model


def detect_anomalies(model, windows_dict, percentile=ANOMALY_PERCENTILE):
    X_train_flat, X_test_flat = windows_dict['X_train_flat'], windows_dict['X_test_flat']
    dates_train, dates_test = windows_dict['dates_train'], windows_dict['dates_test']

    train_pred = model.predict(X_train_flat, verbose=0)
    test_pred = model.predict(X_test_flat, verbose=0)
    train_mse = np.mean(np.square(X_train_flat - train_pred), axis=1)
    test_mse = np.mean(np.square(X_test_flat - test_pred), axis=1)
    threshold = np.percentile(train_mse, percentile)

    anomalies_df = pd.DataFrame({
        'date': np.concatenate([dates_train, dates_test]),
        'reconstruction_error': np.concatenate([train_mse, test_mse]),
        'split': ['train'] * len(dates_train) + ['test'] * len(dates_test),
    })
    anomalies_df['is_anomaly'] = anomalies_df['reconstruction_error'] > threshold
    return anomalies_df, threshold


def run_pipeline(data, ticker):
    df_feat = data[ticker]
    windows = prepare_windows(df_feat)
    model = train_autoencoder(windows)
    anomalies_df, threshold = detect_anomalies(model, windows)
    return {'anomalies_df': anomalies_df, 'threshold': threshold}


# --------------------------------------------------------------------------
# 7. CRUCE DE ANOMALÍAS CON COMUNICACIONES (5.5.3)
# --------------------------------------------------------------------------

def cruce_anomalias_multiactivo(results, df_semantico):
    all_anomaly_dates = []
    for ticker, res in results.items():
        dates = res['anomalies_df'].loc[res['anomalies_df']['is_anomaly'], 'date']
        for d in dates:
            all_anomaly_dates.append({'date': pd.Timestamp(d).normalize(), 'ticker': ticker})
    anomaly_dates_df = pd.DataFrame(all_anomaly_dates)
    if anomaly_dates_df.empty:
        return pd.DataFrame(), 0.0, 0.0

    coincidencias = anomaly_dates_df.groupby('date')['ticker'].agg(lambda x: sorted(set(x))).reset_index()
    coincidencias['n_activos'] = coincidencias['ticker'].apply(len)

    comunicaciones_553 = df_semantico[df_semantico['MARKET_IMPACT'] == 1][['TIMESTAMP']].rename(columns={'TIMESTAMP': 'date'})
    comunicaciones_553['date'] = comunicaciones_553['date'].dt.normalize()
    comm_dates_set = set(comunicaciones_553['date'])

    coincidencias['tiene_comunicacion'] = coincidencias['date'].isin(comm_dates_set)
    pct_con_comm = coincidencias['tiene_comunicacion'].mean() * 100

    all_eval_dates = pd.Series(pd.concat([res['anomalies_df']['date'] for res in results.values()])).dt.normalize().unique()
    pct_base = pd.Series(pd.to_datetime(all_eval_dates)).isin(comm_dates_set).mean() * 100

    return coincidencias.sort_values('n_activos', ascending=False), pct_con_comm, pct_base


def cruce_anomalias_por_activo(data, results, df_semantico):
    filas = []
    for ticker, res in results.items():
        anomalias_ticker = res['anomalies_df']
        fechas_anomalas = set(pd.to_datetime(anomalias_ticker.loc[anomalias_ticker['is_anomaly'], 'date']).dt.normalize())
        todas_fechas_ticker = set(pd.to_datetime(anomalias_ticker['date']).dt.normalize())
        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico)
        fechas_comm_ticker = set(pd.to_datetime(comms_ticker['TIMESTAMP']).dt.normalize())

        if len(fechas_anomalas) == 0:
            continue
        pct_anomalas_con_comm = len(fechas_anomalas & fechas_comm_ticker) / len(fechas_anomalas) * 100
        pct_base_ticker = len(todas_fechas_ticker & fechas_comm_ticker) / len(todas_fechas_ticker) * 100 if todas_fechas_ticker else 0
        filas.append({
            'ticker': ticker, 'n_anomalias': len(fechas_anomalas),
            'pct_anomalias_con_comunicacion': round(pct_anomalas_con_comm, 1),
            'tasa_base_ticker': round(pct_base_ticker, 1),
            'diferencia': round(pct_anomalas_con_comm - pct_base_ticker, 1),
        })
    return pd.DataFrame(filas).sort_values('diferencia', ascending=False)


# --------------------------------------------------------------------------
# 8. BLOQUE 6 (exploratorio): eventos importantes por umbral + AUC
# --------------------------------------------------------------------------

def identificar_eventos_importantes(data, ticker, umbral_desv=UMBRAL_EVENTO_DESV):
    df_t = data[ticker].copy()
    media_ret, std_ret = df_t['log_return'].mean(), df_t['log_return'].std()
    media_vol, std_vol = df_t['volatility_20d'].mean(), df_t['volatility_20d'].std()
    df_t['z_return'] = (df_t['log_return'] - media_ret) / std_ret
    df_t['z_volatility'] = (df_t['volatility_20d'] - media_vol) / std_vol
    eventos = df_t[(df_t['z_return'].abs() > umbral_desv) | (df_t['z_volatility'] > umbral_desv)].copy()
    return eventos[['date', 'log_return', 'z_return', 'volatility_20d', 'z_volatility']]


def auc_intensidad_y_combinado(data, df_semantico, eventos_por_activo):
    """AUC simple (comunicación extrema sí/no) y AUC con validación cruzada
    temporal (combinando intensidad máxima, media y nº de comunicaciones)."""
    filas_simple, filas_cv = [], []

    for ticker in data.keys():
        df_t = data[ticker].sort_values('date').reset_index(drop=True)
        fechas_trading = df_t['date'].tolist()
        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico).copy()
        comms_ticker['fecha'] = pd.to_datetime(comms_ticker['TIMESTAMP']).dt.normalize()
        eventos_set = set(pd.to_datetime(eventos_por_activo[ticker]['date']))

        filas_features = []
        for i in range(1, len(fechas_trading)):
            fecha_actual, fecha_sesion_anterior = fechas_trading[i], fechas_trading[i - 1]
            ventana = comms_ticker[(comms_ticker['fecha'] >= fecha_sesion_anterior) & (comms_ticker['fecha'] <= fecha_actual)]
            intensidad_max = ventana['sentimiento_continuo'].abs().max() if len(ventana) > 0 else 0.0
            filas_features.append({
                'tiene_extrema': int(intensidad_max > UMBRAL_INTENSIDAD),
                'intensidad_max': intensidad_max,
                'intensidad_media': ventana['sentimiento_continuo'].abs().mean() if len(ventana) > 0 else 0.0,
                'n_comunicaciones': len(ventana),
                'es_evento': int(fecha_actual in eventos_set),
            })

        df_features = pd.DataFrame(filas_features)
        if df_features['es_evento'].nunique() > 1:
            auc_simple = roc_auc_score(df_features['es_evento'], df_features['tiene_extrema'])
        else:
            auc_simple = np.nan
        filas_simple.append({'ticker': ticker, 'auc_simple': auc_simple, 'n_eventos': df_features['es_evento'].sum()})

        X = df_features[['intensidad_max', 'intensidad_media', 'n_comunicaciones']]
        y = df_features['es_evento']
        modelo = LogisticRegression(class_weight='balanced')
        tscv = TimeSeriesSplit(n_splits=3)
        aucs_validos = []
        for train_idx, test_idx in tscv.split(X):
            if y.iloc[test_idx].nunique() < 2:
                continue
            modelo.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = modelo.predict_proba(X.iloc[test_idx])[:, 1]
            aucs_validos.append(roc_auc_score(y.iloc[test_idx], pred))

        if aucs_validos:
            filas_cv.append({'ticker': ticker, 'auc_cv': np.mean(aucs_validos), 'auc_std': np.std(aucs_validos), 'n_splits_validos': len(aucs_validos)})
        else:
            filas_cv.append({'ticker': ticker, 'auc_cv': np.nan, 'auc_std': np.nan, 'n_splits_validos': 0})

    return pd.DataFrame(filas_simple).sort_values('auc_simple', ascending=False), \
        pd.DataFrame(filas_cv).sort_values('auc_cv', ascending=False)


def comunicaciones_asociadas_a_eventos_resumen(data, df_semantico, ticker, eventos_df):
    """Devuelve solo el resumen agregado (% de eventos con comunicación
    precedente) — no el detalle fila a fila, que no aporta valor en un log
    de ejecución automático."""
    df_t = data[ticker].sort_values('date').reset_index(drop=True)
    fechas_trading = df_t['date'].tolist()
    comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico).copy()
    comms_ticker['fecha'] = pd.to_datetime(comms_ticker['TIMESTAMP']).dt.normalize()

    n_eventos_con_comm = 0
    for _, evento in eventos_df.iterrows():
        fecha_evento = pd.Timestamp(evento['date']).normalize()
        if fecha_evento not in fechas_trading:
            continue
        idx_evento = fechas_trading.index(fecha_evento)
        if idx_evento == 0:
            continue
        fecha_sesion_anterior = fechas_trading[idx_evento - 1]
        n_comm = len(comms_ticker[(comms_ticker['fecha'] >= fecha_sesion_anterior) & (comms_ticker['fecha'] <= fecha_evento)])
        if n_comm > 0:
            n_eventos_con_comm += 1

    n_con_comunicacion_base = 0
    for i in range(1, len(fechas_trading)):
        fecha_actual, fecha_sesion_anterior = fechas_trading[i], fechas_trading[i - 1]
        if ((comms_ticker['fecha'] >= fecha_sesion_anterior) & (comms_ticker['fecha'] <= fecha_actual)).any():
            n_con_comunicacion_base += 1
    tasa_base = n_con_comunicacion_base / (len(fechas_trading) - 1) * 100 if len(fechas_trading) > 1 else 0

    pct_eventos = (n_eventos_con_comm / len(eventos_df) * 100) if len(eventos_df) > 0 else 0
    return {
        'ticker': ticker, 'n_eventos_importantes': len(eventos_df),
        'pct_eventos_con_comunicacion': round(pct_eventos, 1),
        'tasa_base': round(tasa_base, 1), 'diferencia': round(pct_eventos - tasa_base, 1),
    }


# --------------------------------------------------------------------------
# 9. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------

def main():
    drive_service = get_drive_service()

    # 9.1 Descargar datos financieros (módulo 1) y semánticos (módulo 2)
    carpeta_fin_id = resolver_carpeta_drive(drive_service, DRIVE_FINANCIERO_PATH)
    data = {}
    for ticker, filename in ASSETS.items():
        file_id = buscar_archivo(drive_service, carpeta_fin_id, filename)
        if not file_id:
            print(f"[{ticker}] Archivo no encontrado en Drive: {filename} — se omite")
            continue
        local_path = LOCAL_DIR / filename
        descargar_archivo(drive_service, file_id, local_path)
        df = load_and_clean(local_path, ticker)
        data[ticker] = compute_features(df)
    print(f"Datos financieros cargados: {len(data)} activos.")

    carpeta_sem_id = resolver_carpeta_drive(drive_service, DRIVE_SEMANTICO_PATH)
    sem_file_id = buscar_archivo(drive_service, carpeta_sem_id, SEMANTICO_FILENAME)
    if not sem_file_id:
        raise FileNotFoundError(f"No se encontró '{SEMANTICO_FILENAME}' en Drive (¿se ha ejecutado el módulo 2?).")
    sem_local_path = LOCAL_DIR / SEMANTICO_FILENAME
    descargar_archivo(drive_service, sem_file_id, sem_local_path)
    df_semantico = pd.read_csv(sem_local_path, sep=';', encoding='utf-8-sig')
    df_semantico['TIMESTAMP'] = pd.to_datetime(df_semantico['TIMESTAMP'], format='mixed', errors='coerce')
    df_semantico['sentimiento_continuo'] = df_semantico['prob_positive'] - df_semantico['prob_negative']
    print(f"Corpus semántico cargado: {len(df_semantico)} comunicaciones.")

    # 9.2 Estudio de eventos (5.1) y pre/post (5.2-5.3)
    print("Ejecutando estudio de eventos (5.1)...")
    event_study_summary = run_event_study_all(data, df_semantico)
    print("Ejecutando comparación pre/post (5.2-5.3)...")
    pre_post_summary = run_pre_post_all(data, df_semantico)

    # 9.3 Correlación semántica (5.4)
    print("Ejecutando correlación sentimiento-retorno (5.4)...")
    semantic_corr_summary = run_semantic_corr_all_1d(data, df_semantico)

    # 9.4 Autoencoders (5.5.1-5.5.2)
    print("Entrenando autoencoders por activo (5.5.1-5.5.2)...")
    results = {}
    for ticker in data.keys():
        if data[ticker].empty:
            continue
        results[ticker] = run_pipeline(data, ticker)
        n_anom = results[ticker]['anomalies_df']['is_anomaly'].sum()
        print(f"  [{ticker}] {n_anom} anomalías detectadas.")

    # 9.5 Cruce de anomalías con comunicaciones (5.5.3)
    print("Cruzando anomalías con comunicaciones (5.5.3)...")
    coincidencias, pct_multiactivo, pct_base_multiactivo = cruce_anomalias_multiactivo(results, df_semantico)
    cruce_por_activo = cruce_anomalias_por_activo(data, results, df_semantico)

    # 9.6 Bloque exploratorio: eventos importantes + AUC
    print("Identificando eventos importantes por umbral y calculando AUC (bloque 6)...")
    eventos_por_activo = {ticker: identificar_eventos_importantes(data, ticker) for ticker in data.keys()}
    resumen_eventos_comm = pd.DataFrame([
        comunicaciones_asociadas_a_eventos_resumen(data, df_semantico, ticker, eventos_por_activo[ticker])
        for ticker in data.keys()
    ])
    auc_simple_df, auc_cv_df = auc_intensidad_y_combinado(data, df_semantico, eventos_por_activo)

    # 9.7 Guardar informes de resultados en Drive (para trazabilidad de cada ejecución)
    carpeta_out_id = resolver_carpeta_drive(drive_service, DRIVE_OUTPUT_PATH)
    informes = {
        "informe_event_study.csv": event_study_summary,
        "informe_pre_post.csv": pre_post_summary,
        "informe_correlacion_semantica.csv": semantic_corr_summary,
        "informe_cruce_anomalias_multiactivo.csv": coincidencias,
        "informe_cruce_anomalias_por_activo.csv": cruce_por_activo,
        "informe_eventos_importantes_vs_comunicaciones.csv": resumen_eventos_comm,
        "informe_auc_simple.csv": auc_simple_df,
        "informe_auc_validacion_cruzada.csv": auc_cv_df,
    }
    for nombre, df_informe in informes.items():
        ruta = LOCAL_DIR / nombre
        df_informe.to_csv(ruta, index=False)
        subir_archivo(drive_service, ruta, carpeta_out_id, nombre)

    print(f"\nResumen: {pct_multiactivo:.1f}% de anomalías multi-activo coinciden con comunicaciones "
          f"(tasa base: {pct_base_multiactivo:.1f}%).")

    # 9.8 Dataset final para el capítulo 6 (activos fijos: ACTIVOS_CON_EVIDENCIA)
    print("Generando el dataset consolidado para el capítulo 6...")
    frames = []
    for ticker in ACTIVOS_CON_EVIDENCIA:
        if ticker not in data:
            print(f"  Aviso: '{ticker}' no está disponible en los datos financieros actuales, se omite.")
            continue

        df_t = data[ticker][['date', 'ticker', 'log_return', 'volatility_20d', 'volume_zscore_20d']].copy()

        if ticker in results:
            anom = results[ticker]['anomalies_df'][['date', 'is_anomaly']]
            df_t = df_t.merge(anom, on='date', how='left')
            df_t['is_anomaly'] = np.where(df_t['is_anomaly'].isna(), False, df_t['is_anomaly']).astype(bool)
        else:
            df_t['is_anomaly'] = False

        fechas_evento_importante = set(pd.to_datetime(eventos_por_activo[ticker]['date']))
        df_t['evento_importante'] = pd.to_datetime(df_t['date']).isin(fechas_evento_importante)

        comms_ticker = filtrar_comunicaciones_relevantes(ticker, df_semantico).copy()
        comms_ticker['date'] = pd.to_datetime(comms_ticker['TIMESTAMP']).dt.normalize()
        comunicaciones_diarias = comms_ticker.groupby('date').agg(
            sentiment=('sentimiento_continuo', 'mean'),
            intensidad_max=('sentimiento_continuo', lambda x: x.abs().max()),
            intensidad_media=('sentimiento_continuo', lambda x: x.abs().mean()),
            n_comunicaciones=('sentimiento_continuo', 'count'),
            topic_category=('IMPACT_CATEGORIES', lambda x: '|'.join(sorted(set(x.dropna())))),
        ).reset_index()

        df_t = df_t.merge(comunicaciones_diarias, on='date', how='left')
        df_t['n_comunicaciones'] = df_t['n_comunicaciones'].fillna(0).astype(int)
        df_t['intensidad_max'] = df_t['intensidad_max'].fillna(0.0)
        df_t['intensidad_media'] = df_t['intensidad_media'].fillna(0.0)

        df_t = df_t.sort_values('date').reset_index(drop=True)
        df_t['target_log_return_t1'] = df_t['log_return'].shift(-1)
        frames.append(df_t)

    dataset_capitulo6 = pd.concat(frames, ignore_index=True)
    ruta_salida = LOCAL_DIR / OUTPUT_FILENAME
    dataset_capitulo6.to_csv(ruta_salida, index=False)
    subir_archivo(drive_service, ruta_salida, carpeta_out_id, OUTPUT_FILENAME)

    print(f"Guardado: {OUTPUT_FILENAME} ({len(dataset_capitulo6)} filas, "
          f"{dataset_capitulo6['ticker'].nunique()} activos: {', '.join(ACTIVOS_CON_EVIDENCIA)})")
    print("Análisis del impacto en los mercados completado.")


if __name__ == "__main__":
    main()
