"""
Ingesta de datos financieros (TFM - sección 8.2)
Derivado de 03_obtencion_datos_financieros.ipynb

Descarga OHLCV diario de los activos seleccionados (yfinance) y enriquece
Bitcoin/Ethereum con datos intradía de Binance (hora de máximo/mínimo,
volumen detallado y ratio de compra agresiva). Guarda un CSV por activo,
tanto en local (para depuración) como en la carpeta compartida de Drive.

Pensado para ejecutarse sin intervención humana (GitHub Actions).
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

ASSETS = [
    {"name": "S&P 500",        "ticker": "^GSPC",   "asset_class": "index",            "source": "yfinance"},
    {"name": "Nasdaq",         "ticker": "^IXIC",   "asset_class": "index",            "source": "yfinance"},
    {"name": "Tesla",          "ticker": "TSLA",    "asset_class": "equity",           "source": "yfinance"},
    {"name": "ETF Defensa",    "ticker": "ITA",     "asset_class": "etf_sector",       "source": "yfinance"},
    {"name": "ETF Salud",      "ticker": "XLV",     "asset_class": "etf_sector",       "source": "yfinance"},
    {"name": "ETF Energía",    "ticker": "XLE",     "asset_class": "etf_sector",       "source": "yfinance"},
    {"name": "ETF Financiero", "ticker": "XLF",     "asset_class": "etf_sector",       "source": "yfinance"},
    {"name": "Bitcoin",        "ticker": "BTC-USD", "asset_class": "crypto",           "source": "yfinance"},
    {"name": "Ethereum",       "ticker": "ETH-USD", "asset_class": "crypto",           "source": "yfinance"},
    {"name": "Oro",            "ticker": "GC=F",    "asset_class": "commodity",        "source": "yfinance"},
    {"name": "VIX (control)",  "ticker": "^VIX",    "asset_class": "volatility_index", "source": "yfinance"},
]

CRYPTO_BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

# MODE controla cuánto histórico se pide:
#   "incremental" (por defecto): solo los últimos LOOKBACK_DAYS días, para las
#       ejecuciones automáticas diarias — rápido y no satura las APIs externas.
#   "backfill": el histórico completo desde BACKFILL_START_DATE — se ejecuta
#       UNA SOLA VEZ a mano (o desde "Run workflow" con este modo) para poblar
#       Drive por primera vez. No forma parte de la ejecución programada diaria.
MODE = os.environ.get("MODE", "incremental")
LOOKBACK_DAYS = 5  # margen de seguridad: si un día falla, el siguiente lo recupera
BACKFILL_START_DATE = "2024-11-01"  # alineado con FECHA_INICIO del pipeline de texto

_today = datetime.now(timezone.utc).date()
START_DATE = (
    BACKFILL_START_DATE if MODE == "backfill"
    else (_today - pd.Timedelta(days=LOOKBACK_DAYS)).isoformat()
)
END_DATE = _today.isoformat()

LOCAL_DIR = Path("data_raw_financiero")  # carpeta temporal del runner de GitHub Actions

DRIVE_PARENT_PATH = ["TFM DATA SCIENCE", "03. data"]
DRIVE_OUTPUT_FOLDER_NAME = "01. RAW - Datos Financieros"

BINANCE_INTERVAL = "1d"
BINANCE_BASE_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_LIMIT = 1000


def build_http_session() -> requests.Session:
    """Sesión HTTP con reintentos automáticos ante cortes de red puntuales
    (timeouts, errores 429/500-504, o el servidor cerrando la conexión a
    medio TLS-handshake, como el SSLEOFError que puede dar Binance)."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=2,  # espera 2s, 4s, 8s, 16s, 32s entre reintentos
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


HTTP = build_http_session()


# --------------------------------------------------------------------------
# 2. AUTENTICACIÓN CON GOOGLE DRIVE
# --------------------------------------------------------------------------
# Autenticación desatendida con cuenta de servicio: la clave JSON no vive
# en este archivo ni en el repositorio, se lee desde una variable de entorno
# (GDRIVE_SERVICE_ACCOUNT_KEY) que GitHub Actions rellena a partir del
# secreto del mismo nombre en tiempo de ejecución.

from google.oauth2 import service_account
from googleapiclient.discovery import build

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_drive_service():
    """Devuelve un cliente autenticado de la API de Drive (cuenta de servicio)."""
    key_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
    if not key_json:
        raise RuntimeError(
            "No se encontró la variable de entorno GDRIVE_SERVICE_ACCOUNT_KEY. "
            "En local, expórtala tú mismo antes de probar el script; "
            "en GitHub Actions la rellena el secreto del mismo nombre."
        )

    key_info = json.loads(key_json)
    credentials = service_account.Credentials.from_service_account_info(
        key_info, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=credentials)


# --------------------------------------------------------------------------
# 3. DESCARGA DE PRECIOS (YFINANCE)
# --------------------------------------------------------------------------

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga OHLCV diario desde Yahoo Finance vía yfinance."""
    # yfinance trata 'end' como exclusivo (no incluye ese día), así que se suma
    # un día para que el último día del rango solicitado sí quede incluido.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end_inclusive, interval="1d",
                      progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance no devolvió datos para {ticker}")

    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Date": "DATE", "Open": "OPEN", "High": "HIGH",
        "Low": "LOW", "Close": "CLOSE", "Volume": "VOLUME",
    })
    return df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]


# --------------------------------------------------------------------------
# 4. UTILIDADES DE DRIVE
# --------------------------------------------------------------------------

def resolver_carpeta_drive(drive_service, partes_ruta: list, crear_si_falta: bool = True) -> str:
    """Navega una ruta de carpetas en Drive y devuelve el ID de la carpeta final."""
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
            metadata = {
                "name": nombre,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            carpeta = drive_service.files().create(body=metadata, fields="id").execute()
            parent_id = carpeta["id"]
            print(f"  Carpeta creada: {nombre}")
        else:
            raise FileNotFoundError(
                f"No se encontró la carpeta '{nombre}' en Drive "
                f"(ruta: {' / '.join(partes_ruta[:i + 1])}). Comprueba que tienes acceso."
            )
    return parent_id


def subir_o_actualizar_archivo(drive_service, ruta_local: str, carpeta_drive_id: str, nombre_archivo: str = None) -> str:
    """Sube un archivo a Drive; si ya existe uno con el mismo nombre en esa carpeta, lo actualiza."""
    from googleapiclient.http import MediaFileUpload

    nombre_archivo = nombre_archivo or Path(ruta_local).name
    query = f"name = '{nombre_archivo}' and '{carpeta_drive_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
    existentes = resultado.get("files", [])

    media = MediaFileUpload(ruta_local, resumable=True)

    if existentes:
        file_id = existentes[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        return file_id
    else:
        metadata = {"name": nombre_archivo, "parents": [carpeta_drive_id]}
        archivo = drive_service.files().create(body=metadata, media_body=media, fields="id").execute()
        return archivo["id"]


def descargar_csv_existente(drive_service, carpeta_drive_id: str, nombre_archivo: str) -> pd.DataFrame | None:
    """Si ya existe un CSV con ese nombre en Drive, lo descarga y lo devuelve como
    DataFrame. Si no existe (primera vez, o modo backfill), devuelve None."""
    from googleapiclient.http import MediaIoBaseDownload
    import io

    query = f"name = '{nombre_archivo}' and '{carpeta_drive_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
    encontrados = resultado.get("files", [])
    if not encontrados:
        return None

    file_id = encontrados[0]["id"]
    request = drive_service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buffer.seek(0)
    return pd.read_csv(buffer, parse_dates=["DATE"])


def fusionar_con_existente(df_nuevo: pd.DataFrame, df_existente: pd.DataFrame | None) -> pd.DataFrame:
    """Combina los datos nuevos con los ya guardados, sin duplicar fechas.
    Si una fecha aparece en ambos, gana la versión nueva (por si se corrigió)."""
    if df_existente is None or df_existente.empty:
        return df_nuevo.sort_values("DATE").reset_index(drop=True)

    combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    combinado = combinado.drop_duplicates(subset="DATE", keep="last")
    return combinado.sort_values("DATE").reset_index(drop=True)


def save_dataset(drive_service, df: pd.DataFrame, ticker: str, asset_class: str,
                  source: str, granularity: str, local_dir: Path,
                  drive_folder_id: str) -> str:
    """Añade metadatos estándar, fusiona con lo que ya hubiera en Drive
    (modo incremental) y guarda el CSV resultante en local y en Drive."""
    df = df.copy()

    if "VOLUME" in df.columns:
        new_name = "VOLUME_USD" if asset_class == "crypto" else "VOLUME_UNITS"
        df = df.rename(columns={"VOLUME": new_name})

    df["TICKER"] = ticker
    df["ASSET_CLASS"] = asset_class
    df["SOURCE"] = source
    df["GRANULARITY"] = granularity

    local_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "").replace("/", "_")
    nombre_archivo = f"{safe_name}.csv"
    local_path = local_dir / nombre_archivo

    if MODE == "incremental":
        df_existente = descargar_csv_existente(drive_service, drive_folder_id, nombre_archivo)
        df = fusionar_con_existente(df, df_existente)

    df.to_csv(local_path, index=False)
    subir_o_actualizar_archivo(drive_service, str(local_path), drive_folder_id, nombre_archivo)

    print(f"Guardado: {nombre_archivo} ({len(df)} filas) — local y Drive")
    return str(local_path)


# --------------------------------------------------------------------------
# 5. ENRIQUECIMIENTO CON BINANCE (SOLO CRIPTOACTIVOS)
# --------------------------------------------------------------------------

def fetch_binance_volume_detail(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Descarga velas de Binance con detalle de volumen y ratio de compra agresiva."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": BINANCE_LIMIT,
        }
        resp = HTTP.get(BINANCE_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        cursor = rows[-1][0] + 1
        time.sleep(0.5)

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume_btc",
        "close_time", "volume_usd", "trades",
        "taker_buy_volume_btc", "taker_buy_volume_usd", "ignore",
    ])
    df["DATE"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["volume_btc", "volume_usd", "taker_buy_volume_btc", "taker_buy_volume_usd"]:
        df[col] = df[col].astype(float)

    df["TAKER_BUY_RATIO"] = df["taker_buy_volume_btc"] / df["volume_btc"]
    df = df.rename(columns={
        "volume_btc": "VOLUME_BTC", "volume_usd": "VOLUME_USD",
        "taker_buy_volume_btc": "TAKER_BUY_VOLUME_BTC",
    })

    return df[["DATE", "VOLUME_BTC", "VOLUME_USD", "TAKER_BUY_VOLUME_BTC", "TAKER_BUY_RATIO"]]


def find_daily_high_low_time(symbol: str, date: str) -> dict:
    """Para un día concreto, devuelve la hora exacta (UTC) del máximo y del mínimo."""
    day_start = pd.Timestamp(date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)

    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": int(day_start.timestamp() * 1000),
        "endTime": int(day_end.timestamp() * 1000),
        "limit": 1000,
    }
    resp = HTTP.get(BINANCE_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        return {"DATE": date, "HIGH_TIME": None, "LOW_TIME": None}

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    high_row = df.loc[df["high"].idxmax()]
    low_row = df.loc[df["low"].idxmin()]

    return {
        "DATE": date,
        "HIGH_TIME": high_row["timestamp"].strftime("%H:%M UTC"),
        "LOW_TIME": low_row["timestamp"].strftime("%H:%M UTC"),
    }


def fetch_high_low_times_range(symbol: str, start_date: str, end_date: str, verbose_every: int = 30) -> pd.DataFrame:
    """Para un rango de fechas, calcula día a día la hora exacta del máximo y del mínimo."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    results = []

    for i, day in enumerate(dates):
        date_str = day.strftime("%Y-%m-%d")
        try:
            r = find_daily_high_low_time(symbol, date_str)
        except Exception as e:
            r = {"DATE": date_str, "HIGH_TIME": None, "LOW_TIME": None}
            print(f"  aviso: fallo en {date_str} ({symbol}): {e}")
        results.append(r)
        time.sleep(0.5)
        if (i + 1) % verbose_every == 0:
            print(f"  {symbol}: procesados {i + 1}/{len(dates)} días...")

    return pd.DataFrame(results)


def add_volume_estimates(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """Añade una columna de volumen estimado en la unidad complementaria."""
    df = df.copy()
    if asset_class == "crypto":
        df["VOLUME_UNITS_EST"] = df["VOLUME_USD"] / df["CLOSE"]
    else:
        df["VOLUME_USD_EST"] = df["VOLUME_UNITS"] * df["CLOSE"]
    return df


# --------------------------------------------------------------------------
# 6. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------

def main():
    print(f"Iniciando ingesta financiera: {START_DATE} -> {END_DATE}")

    drive_service = get_drive_service()
    drive_folder_id = resolver_carpeta_drive(
        drive_service, DRIVE_PARENT_PATH + [DRIVE_OUTPUT_FOLDER_NAME]
    )

    # 6.1 Descarga base (todos los activos)
    for asset in ASSETS:
        ticker = asset["ticker"]
        df = fetch_yfinance(ticker, START_DATE, END_DATE)
        save_dataset(drive_service, df, ticker, asset["asset_class"], asset["source"],
                     "1d", LOCAL_DIR, drive_folder_id)

    # 6.2 Enriquecimiento: hora exacta de máximo/mínimo (solo Bitcoin y Ethereum)
    for asset in ASSETS:
        ticker = asset["ticker"]
        if ticker not in CRYPTO_BINANCE_SYMBOLS:
            continue

        binance_symbol = CRYPTO_BINANCE_SYMBOLS[ticker]
        print(f"Calculando hora de máximo/mínimo diario para {ticker} ({binance_symbol})...")
        df_hl = fetch_high_low_times_range(binance_symbol, START_DATE, END_DATE)

        safe_name = ticker.replace("^", "").replace("/", "_")
        local_path = LOCAL_DIR / f"{safe_name}.csv"
        df_price = pd.read_csv(local_path, parse_dates=["DATE"])
        df_price["DATE_KEY"] = df_price["DATE"].dt.strftime("%Y-%m-%d")

        df_merged = df_price.merge(
            df_hl[["DATE", "HIGH_TIME", "LOW_TIME"]],
            left_on="DATE_KEY", right_on="DATE", how="left", suffixes=("", "_HL")
        ).drop(columns=["DATE_KEY", "DATE_HL"], errors="ignore")

        # En ejecuciones sucesivas HIGH_TIME/LOW_TIME ya existían: nos quedamos
        # con el valor nuevo si lo hay, y si no, con el que ya estaba guardado.
        for col in ["HIGH_TIME", "LOW_TIME"]:
            col_nueva = f"{col}_HL"
            if col_nueva in df_merged.columns:
                df_merged[col] = df_merged[col_nueva].combine_first(df_merged[col])
                df_merged = df_merged.drop(columns=[col_nueva])

        df_merged.to_csv(local_path, index=False)
        subir_o_actualizar_archivo(drive_service, str(local_path), drive_folder_id, f"{safe_name}.csv")
        print(f"  Actualizado: {safe_name}.csv (+{df_merged['HIGH_TIME'].notna().sum()} días con hora exacta)")

    # 6.3 Enriquecimiento: volumen estimado + ratio de compra agresiva
    for asset in ASSETS:
        ticker = asset["ticker"]
        asset_class = asset["asset_class"]
        safe_name = ticker.replace("^", "").replace("/", "_")
        local_path = LOCAL_DIR / f"{safe_name}.csv"

        df_price = pd.read_csv(local_path, parse_dates=["DATE"])
        df_price = add_volume_estimates(df_price, asset_class)

        if ticker in CRYPTO_BINANCE_SYMBOLS:
            binance_symbol = CRYPTO_BINANCE_SYMBOLS[ticker]
            df_vol = fetch_binance_volume_detail(binance_symbol, BINANCE_INTERVAL, START_DATE, END_DATE)
            df_vol["DATE"] = pd.to_datetime(df_vol["DATE"]).dt.tz_localize(None).dt.normalize()
            df_price["DATE_KEY"] = pd.to_datetime(df_price["DATE"]).dt.tz_localize(None).dt.normalize()
            df_price = df_price.merge(
                df_vol[["DATE", "TAKER_BUY_RATIO"]],
                left_on="DATE_KEY", right_on="DATE", how="left", suffixes=("", "_VOL")
            ).drop(columns=["DATE_KEY", "DATE_VOL"], errors="ignore")

            if "TAKER_BUY_RATIO_VOL" in df_price.columns:
                df_price["TAKER_BUY_RATIO"] = df_price["TAKER_BUY_RATIO_VOL"].combine_first(
                    df_price["TAKER_BUY_RATIO"]
                )
                df_price = df_price.drop(columns=["TAKER_BUY_RATIO_VOL"])

        df_price.to_csv(local_path, index=False)
        subir_o_actualizar_archivo(drive_service, str(local_path), drive_folder_id, f"{safe_name}.csv")
        extra = " + TAKER_BUY_RATIO" if ticker in CRYPTO_BINANCE_SYMBOLS else ""
        print(f"Actualizado: {safe_name}.csv (+volumen estimado{extra})")

    print("Ingesta financiera completada.")


if __name__ == "__main__":
    main()
