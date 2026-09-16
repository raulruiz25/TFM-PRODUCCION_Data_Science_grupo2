"""
Limpieza y transformación de datos textuales (TFM - sección 8.2, apartados 3.5/3.6 del índice)
Derivado de 03__Limpieza_y_Transformación_de_datos.ipynb

Descarga las 4 fuentes en bruto de comunicaciones (Musk en 2 JSON, Trump en
CSV, FED en CSV) desde Drive, las une en un esquema común, corrige mojibake,
deduplica, filtra al horizonte temporal del TFM, clasifica cada comunicación
por impacto de mercado (método de diccionario de palabras clave) y guarda el
corpus unificado resultante (dataset_unificado.csv) en Drive.

Es el primer eslabón de la cadena de texto: su salida es la entrada del
módulo de análisis semántico. Las 4 fuentes en bruto son ficheros estáticos
subidos manualmente a Drive (no hay ingesta automática de comunicaciones
nuevas, ver limitación documentada en el módulo de análisis semántico), así
que este módulo no es incremental: recalcula el corpus completo en cada
ejecución a partir de los mismos ficheros de origen.

Pensado para ejecutarse sin intervención humana (GitHub Actions).
"""

import io
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # resultados reproducibles

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

FECHA_INICIO = "2024-11-01"
FECHA_FIN = "2026-06-30"

LOCAL_DIR = Path("data_limpieza_textual")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

NOMBRE_MUSK_BASE = "elonmusk-ALL-2025-12-10.json"
NOMBRE_MUSK_EXTRA = "elonmusk_tweets_2025-11-02T01-12-31-543Z.json"
NOMBRE_TRUMP = "Trump_truth_archive.csv"
NOMBRE_FED = "fed_statements.csv"

DRIVE_INPUT_PATH = ["TFM DATA SCIENCE", "03. data", "02. RAW - Datos Textuales"]
DRIVE_OUTPUT_PATH = ["TFM DATA SCIENCE", "03. data", "04. PROCESSED - Datos Textuales"]
OUTPUT_FILENAME = "dataset_unificado.csv"

FINAL_COLUMNS = [
    "SOURCE_FILE", "PLATFORM", "AUTHOR", "DATE", "HOUR", "TEXT",
    "REPLIES", "REPOSTS", "LIKES", "VIEWS", "QUOTES", "BOOKMARKS",
    "ID", "URL", "MEDIA", "CONVERSATION_ID", "LANG_SOURCE",
    "HAS_DATA_LOSS", "CONTENT_SCRAPE_FAILURE",
]

CONFIG_MUSK = {
    "archivo": "elonmusk-ALL-2025-12-10.json (+ enriquecimiento)",
    "red_social": "TWITTER",
    "autor": "MUSK",
    "columna_datetime": "created_at",
    "tiene_hora": True,
    "hora_defecto": None,
    "columnas": {
        "TEXT": "text",
        "REPLIES": "stats.replies",
        "REPOSTS": "stats.retweets",
        "LIKES": "stats.likes",
        "VIEWS": "stats.views",
        "QUOTES": "stats.quotes",
        "BOOKMARKS": "stats.bookmarks",
        "ID": "id",
        "URL": "url",
        "MEDIA": "media",
        "CONVERSATION_ID": "conversationId",
        "LANG_SOURCE": "lang",
        "HAS_DATA_LOSS": None,
        "CONTENT_SCRAPE_FAILURE": None,
    },
}

CONFIG_TRUMP = {
    "archivo": "Trump_truth_archive.csv",
    "red_social": "TRUMPS TRUTH",
    "autor": "TRUMP",
    "columna_datetime": "created_at",
    "tiene_hora": True,
    "hora_defecto": None,
    "columnas": {
        "TEXT": "content",
        "REPLIES": "replies_count",
        "REPOSTS": "reblogs_count",
        "LIKES": "favourites_count",
        "VIEWS": None,
        "QUOTES": None,
        "BOOKMARKS": None,
        "ID": "id",
        "URL": "url",
        "MEDIA": "media",
        "CONVERSATION_ID": None,
        "LANG_SOURCE": None,
        "HAS_DATA_LOSS": None,
        "CONTENT_SCRAPE_FAILURE": "CONTENT_SCRAPE_FAILURE",
    },
}

CONFIG_FED = {
    "archivo": "fed_statements.csv",
    "red_social": "FED",
    "autor": "FED",
    "columna_datetime": "Date",
    "tiene_hora": False,
    "hora_defecto": "14:30:00",
    "columnas": {
        "TEXT": "Text",
        "REPLIES": None,
        "REPOSTS": None,
        "LIKES": None,
        "VIEWS": None,
        "QUOTES": None,
        "BOOKMARKS": None,
        "ID": None,
        "URL": None,
        "MEDIA": None,
        "CONVERSATION_ID": None,
        "LANG_SOURCE": None,
        "HAS_DATA_LOSS": "HAS_DATA_LOSS",
        "CONTENT_SCRAPE_FAILURE": None,
    },
}

# Clasificación de impacto de mercado por diccionario de palabras clave
# (ver apartado 5.5.4 del TFM para la comparación frente a zero-shot: el
# método de diccionario es el elegido para producción).
CATEGORIAS_IMPACTO_DEF = {
    "EMPRESAS": [
        r"tesla", r"\btsla\b", r"spacex", r"starlink", r"neuralink",
        r"twitter", r"\bx corp\b", r"\bxai\b", r"\bgrok\b",
        r"apple", r"\baapl\b", r"amazon", r"\bamzn\b", r"microsoft", r"\bmsft\b",
        r"google", r"alphabet", r"\bgoogl\b", r"\bmeta\b", r"facebook", r"nvidia", r"\bnvda\b",
        r"intel", r"\bamd\b", r"\btsmc\b", r"taiwan semiconductor", r"boeing", r"lockheed",
        r"goldman", r"jpmorgan", r"\bjp morgan\b", r"blackrock", r"walmart", r"disney",
        r"openai", r"anthropic", r"deepseek", r"\bford\b", r"general motors",
        r"\bipo\b", r"earnings", r"\bmerger\b", r"acquisition", r"bankrupt",
        r"shareholder", r"accionistas", r"\bstocks?\b", r"\bshares\b",
        r"\bbolsa\b", r"wall street", r"\bnasdaq\b", r"\bs&p\b", r"\bdow jones\b",
    ],
    "CRIPTO": [
        r"bitcoin", r"\bbtc\b", r"ethereum", r"\beth\b", r"dogecoin",
        r"crypto", r"cripto", r"blockchain", r"stablecoin", r"\busdt\b", r"\busdc\b",
        r"binance", r"coinbase", r"\bnft\b", r"\bdefi\b", r"\bsolana\b", r"\bxrp\b",
        r"digital asset", r"strategic reserve",
    ],
    "ARANCELES_COMERCIO": [
        r"tariff", r"arancel", r"trade deal", r"trade war", r"guerra comercial",
        r"\bimports?\b", r"\bexports?\b", r"importaci[oó]n", r"exportaci[oó]n",
        r"\bnafta\b", r"\busmca\b", r"\bwto\b", r"\bomc\b", r"trade deficit",
        r"d[ée]ficit comercial", r"embargo", r"sanction", r"sanci[oó]n",
        r"supply chain", r"cadena de suministro", r"\bquotas?\b", r"proteccionis",
    ],
    "GEOPOLITICA": [
        r"\biran\b", r"\bir[aá]n\b", r"israel", r"\bgaza\b", r"hamas", r"hezboll?ah", r"hezbol[aá]",
        r"middle east", r"oriente medio", r"\bsyria\b", r"\bsiria\b", r"\byemen\b", r"houthi",
        r"\bhormuz\b", r"\bred sea\b", r"mar rojo", r"saudi", r"arabia saud[ií]",
        r"venezuela", r"maduro", r"caracas", r"\bpdvsa\b",
        r"\brussia\b", r"\brusia\b", r"\bputin\b", r"ukrain", r"ucrania", r"zelensk",
        r"\bchina\b", r"xi jinping", r"taiwan", r"taiw[aá]n", r"north korea", r"corea del norte",
        r"\bnato\b", r"\botan\b", r"\bwar\b", r"\bguerra\b", r"military strike", r"airstrike",
        r"invasion", r"invasi[oó]n", r"ceasefire", r"alto el fuego", r"missile", r"misil",
    ],
    "POLITICA_MONETARIA": [
        r"federal reserve", r"\bthe fed\b", r"\bfed\b", r"\bfomc\b", r"jerome powell", r"\bpowell\b",
        r"kevin warsh", r"\bwarsh\b",
        r"interest rates?", r"tipos de inter[eé]s", r"rate cut", r"rate hike",
        r"inflation", r"inflaci[oó]n", r"\bcpi\b", r"\bipc\b", r"monetary policy",
        r"pol[ií]tica monetaria", r"quantitative easing", r"money supply", r"basis points?",
        r"\btreasury\b", r"\btreasuries\b", r"\bbonds?\b", r"\bbonos\b", r"\byields?\b",
    ],
    "MACRO_FISCAL": [
        r"\bgdp\b", r"\bpib\b", r"recession", r"recesi[oó]n", r"unemployment", r"desempleo",
        r"jobs report", r"\btaxes?\b", r"\bimpuestos?\b", r"tax cut", r"tax hike",
        r"\bdeficit\b", r"\bd[eé]ficit\b", r"national debt", r"\bdeuda\b", r"debt ceiling",
        r"government shutdown", r"stimulus", r"est[ií]mulo", r"\bbudget\b", r"presupuesto",
        r"\bdoge\b", r"government spending", r"subsid",
    ],
    "ENERGIA_MATERIAS": [
        r"\boil\b", r"petr[oó]leo", r"\bcrude\b", r"\bopec\b", r"\bopep\b", r"\bbarrels?\b",
        r"natural gas", r"gasoline", r"gasolina", r"\bgold\b", r"\boro\b",
        r"\bcopper\b", r"\bcobre\b", r"lithium", r"litio", r"rare earths?", r"tierras raras",
        r"drilling", r"\bpipeline\b", r"energy prices?", r"precio de la energ[ií]a",
        r"\bhormuz\b",
    ],
    "REGULACION": [
        r"\bsec\b", r"\bftc\b", r"\bdoj\b", r"antitrust", r"antimonopolio",
        r"regulation", r"regulaci[oó]n", r"deregulat", r"executive order", r"orden ejecutiva",
        r"lawsuit", r"demanda judicial", r"\bfines?\b", r"\bmultas?\b", r"investigation",
        r"\bban\b", r"prohib", r"compliance", r"subpoena", r"supreme court", r"tribunal supremo",
    ],
}
FUENTES_SIEMPRE_IMPACTO = {"FED"}
CATEGORIAS_COMPILADAS = {
    cat: re.compile("|".join(patrones), flags=re.IGNORECASE)
    for cat, patrones in CATEGORIAS_IMPACTO_DEF.items()
}

# --------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y UTILIDADES DE DRIVE (mismo patrón que los módulos anteriores)
# --------------------------------------------------------------------------

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_drive_service():
    """Devuelve un cliente autenticado de la API de Drive (cuenta de servicio)."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

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


def buscar_archivo_por_nombre(drive_service, carpeta_id: str, nombre: str) -> str:
    """Busca un archivo por nombre dentro de una carpeta de Drive. Devuelve su file_id."""
    query = f"name = '{nombre}' and '{carpeta_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)", pageSize=10).execute()
    archivos = resultado.get("files", [])
    if not archivos:
        raise FileNotFoundError(f"No se encontró '{nombre}' en Drive. Comprueba que ya se subió a la carpeta.")
    return archivos[0]["id"]


def descargar_archivo(drive_service, file_id: str, destino: Path):
    """Descarga un archivo de Drive por su ID a una ruta local."""
    from googleapiclient.http import MediaIoBaseDownload

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destino, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()


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


def leer_csv_drive(drive_service, carpeta_id: str, nombre_archivo: str) -> pd.DataFrame:
    """Busca, descarga y lee un CSV de Drive por nombre, detectando el separador."""
    file_id = buscar_archivo_por_nombre(drive_service, carpeta_id, nombre_archivo)
    destino_local = LOCAL_DIR / nombre_archivo
    descargar_archivo(drive_service, file_id, destino_local)
    return pd.read_csv(destino_local, sep=None, engine="python")


def leer_json_drive(drive_service, carpeta_id: str, nombre_archivo: str):
    """Busca, descarga y lee un JSON de Drive por nombre."""
    file_id = buscar_archivo_por_nombre(drive_service, carpeta_id, nombre_archivo)
    destino_local = LOCAL_DIR / nombre_archivo
    descargar_archivo(drive_service, file_id, destino_local)
    with open(destino_local, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 3. TRANSFORMACIÓN AL ESQUEMA UNIFICADO
# --------------------------------------------------------------------------

def obtener_columna(df: pd.DataFrame, nombre_columna):
    """Devuelve la columna solicitada, o una columna de NaN si no aplica (None) o no existe."""
    if nombre_columna is None:
        return pd.Series(np.nan, index=df.index)
    if nombre_columna not in df.columns:
        raise KeyError(
            f"La columna '{nombre_columna}' no existe en el CSV. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return df[nombre_columna]


def transformar_fuente(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convierte un DataFrame origen al esquema unificado según su configuración."""
    out = pd.DataFrame(index=df.index)
    nombre_archivo = config["archivo"]
    out["SOURCE_FILE"] = nombre_archivo
    out["PLATFORM"] = config["red_social"]
    out["AUTHOR"] = config["autor"]

    columna_origen_dt = obtener_columna(df, config["columna_datetime"])
    fecha_hora = pd.to_datetime(columna_origen_dt, errors="coerce", utc=True).dt.tz_localize(None)
    filas_no_parseadas = fecha_hora.isna().sum() - columna_origen_dt.isna().sum()
    if filas_no_parseadas > 0:
        print(f"  Aviso ({nombre_archivo}): {filas_no_parseadas} fecha(s) no se pudieron interpretar y quedarán vacías.")

    out["DATE"] = fecha_hora.dt.strftime("%d/%m/%Y")

    hora_defecto = config.get("hora_defecto")
    if config["tiene_hora"]:
        out["HOUR"] = fecha_hora.dt.strftime("%H:%M:%S")
        out["_FECHA_DT"] = fecha_hora
    elif hora_defecto is not None:
        out["HOUR"] = hora_defecto
        fecha_str = fecha_hora.dt.strftime("%Y-%m-%d")
        out["_FECHA_DT"] = pd.to_datetime(fecha_str + " " + hora_defecto, errors="coerce")
    else:
        out["HOUR"] = np.nan
        out["_FECHA_DT"] = fecha_hora

    out["_HORA_ASUMIDA"] = not config["tiene_hora"]

    for campo_final, columna_origen in config["columnas"].items():
        if campo_final == "ID":
            serie_id = obtener_columna(df, columna_origen)
            out[campo_final] = serie_id.apply(lambda x: str(int(x)) if pd.notna(x) else None)
        else:
            out[campo_final] = obtener_columna(df, columna_origen)

    return out[FINAL_COLUMNS + ["_FECHA_DT", "_HORA_ASUMIDA"]]


def elegir_fila_correcta_fed(grupo: pd.DataFrame, diferencia_esperada_por_tipo: dict):
    """Entre varias filas de FED con el mismo texto (duplicado), elige la que tiene
    la 'Release Date' más plausible según la diferencia típica observada para ese
    tipo de documento (Statement/Minute)."""
    con_release = grupo[grupo["Release Date"].notna()].copy()
    if len(con_release) == 0:
        return grupo.iloc[[0]]
    if len(con_release) == 1:
        return con_release
    tipo = grupo["Type"].iloc[0]
    objetivo = diferencia_esperada_por_tipo.get(tipo, 0)
    con_release["_diff"] = (pd.to_datetime(con_release["Release Date"]) - pd.to_datetime(con_release["Date"])).dt.days
    con_release["_dist"] = (con_release["_diff"] - objetivo).abs()
    return con_release.sort_values("_dist").iloc[[0]].drop(columns=["_diff", "_dist"])


def normalizar_texto(texto) -> str:
    """Limpia HTML y URLs para que no generen falsos positivos en la clasificación."""
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"https?://\S+", " ", texto)
    return unicodedata.normalize("NFC", texto)


def clasificar_impacto(texto, red_social: str):
    """Devuelve (impacto 0/1, categorías detectadas separadas por '|')."""
    if red_social in FUENTES_SIEMPRE_IMPACTO:
        return 1, "FED_COMUNICADO"

    limpio = normalizar_texto(texto)
    if not limpio.strip():
        return 0, ""

    encontradas = [cat for cat, rx in CATEGORIAS_COMPILADAS.items() if rx.search(limpio)]
    return (1 if encontradas else 0), "|".join(encontradas)


def detectar_idioma(texto, min_palabras: int = 3):
    if pd.isna(texto) or len(str(texto).split()) < min_palabras:
        return None
    try:
        return detect(texto)
    except LangDetectException:
        return None


# --------------------------------------------------------------------------
# 4. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------

def main():
    print(f"Rango de fechas del corpus: {FECHA_INICIO} -> {FECHA_FIN}")
    drive_service = get_drive_service()
    carpeta_input_id = resolver_carpeta_drive(drive_service, DRIVE_INPUT_PATH, crear_si_falta=False)

    # 4.1 Descarga de las 4 fuentes en bruto
    data_musk_base = leer_json_drive(drive_service, carpeta_input_id, NOMBRE_MUSK_BASE)
    data_musk_extra = leer_json_drive(drive_service, carpeta_input_id, NOMBRE_MUSK_EXTRA)
    df_musk_base = pd.json_normalize(data_musk_base)
    df_musk_extra = pd.json_normalize(data_musk_extra)[["id", "conversationId", "lang", "media"]]

    df_trump = leer_csv_drive(drive_service, carpeta_input_id, NOMBRE_TRUMP)
    df_fed = leer_csv_drive(drive_service, carpeta_input_id, NOMBRE_FED)
    print(f"Cargado: Musk base {len(df_musk_base)} filas, Musk enriquecimiento {len(df_musk_extra)} filas, "
          f"Trump {len(df_trump)} filas, FED {len(df_fed)} filas")

    # 4.2 Combinar Musk base + enriquecimiento (dedup del enriquecimiento por id, left join)
    df_musk_extra_dedup = df_musk_extra.drop_duplicates(subset=["id"], keep="first")
    df_musk = df_musk_base.merge(df_musk_extra_dedup, on="id", how="left")
    assert len(df_musk) == len(df_musk_base), (
        f"El merge alteró el número de filas: {len(df_musk)} vs {len(df_musk_base)} esperadas"
    )

    # 4.3 Deduplicar FED por texto exacto, quedándonos con la Release Date más plausible
    fechas_fed_date = pd.to_datetime(df_fed["Date"], errors="coerce", utc=True)
    fechas_fed_release = pd.to_datetime(df_fed["Release Date"], errors="coerce", utc=True)
    df_fed_temp = df_fed.copy()
    df_fed_temp["_diff_dias"] = (fechas_fed_release - fechas_fed_date).dt.days
    diferencia_esperada_por_tipo = df_fed_temp.groupby("Type")["_diff_dias"].median().to_dict()

    dup_fed_texto = df_fed[df_fed.duplicated(subset=["Text"], keep=False)]
    if len(dup_fed_texto) > 0:
        filas_correctas = [
            elegir_fila_correcta_fed(grupo, diferencia_esperada_por_tipo)
            for _, grupo in dup_fed_texto.groupby("Text")
        ]
        df_fed = pd.concat([
            df_fed[~df_fed["Text"].isin(dup_fed_texto["Text"])],
            pd.concat(filas_correctas),
        ]).sort_values("Date").reset_index(drop=True)
    print(f"FED tras deduplicar por texto: {len(df_fed)} filas")

    # 4.4 Corregir mojibake (codificación mal decodificada) en los 3 campos de texto
    df_musk["text"] = df_musk["text"].apply(ftfy.fix_text)
    df_trump["content"] = df_trump["content"].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)
    df_fed["Text"] = df_fed["Text"].apply(ftfy.fix_text)

    # 4.5 Marcar filas con dato irrecuperable (FED) o posible fallo de scraping (Trump)
    df_fed["HAS_DATA_LOSS"] = df_fed["Text"].str.contains("\ufffd", na=False)
    df_trump["CONTENT_SCRAPE_FAILURE"] = df_trump["content"].isna() & df_trump["media"].isna()

    # 4.6 Transformar cada fuente al esquema unificado y concatenar
    df_musk_t = transformar_fuente(df_musk, CONFIG_MUSK)
    df_trump_t = transformar_fuente(df_trump, CONFIG_TRUMP)
    df_fed_t = transformar_fuente(df_fed, CONFIG_FED)
    df_final = pd.concat([df_musk_t, df_trump_t, df_fed_t], ignore_index=True)
    print(f"Corpus combinado: {len(df_final)} filas")

    # 4.7 Filtrar al horizonte temporal del TFM
    filas_antes = len(df_final)
    df_final = df_final[
        (df_final["_FECHA_DT"] >= pd.Timestamp(FECHA_INICIO)) &
        (df_final["_FECHA_DT"] <= pd.Timestamp(FECHA_FIN) + pd.Timedelta(hours=23, minutes=59, seconds=59))
    ].reset_index(drop=True)
    print(f"Filas dentro del horizonte ({FECHA_INICIO} -> {FECHA_FIN}): {len(df_final)} de {filas_antes}")

    # 4.8 Clasificación de impacto de mercado (método de diccionario)
    resultado = df_final.apply(
        lambda fila: clasificar_impacto(fila["TEXT"], fila["PLATFORM"]), axis=1
    )
    df_final["MARKET_IMPACT"] = [r[0] for r in resultado]
    df_final["IMPACT_CATEGORIES"] = [r[1] for r in resultado]

    # 4.9 Columnas derivadas finales
    df_final["HOUR_MISSING"] = df_final["_HORA_ASUMIDA"]
    df_final["TIMESTAMP"] = df_final["_FECHA_DT"]
    df_final["HAS_TEXT"] = ~(df_final["TEXT"].isna() | (df_final["TEXT"].fillna("").str.strip() == ""))
    df_final["TEXT_LENGTH_CHARS"] = df_final["TEXT"].str.len()
    df_final["TEXT_LENGTH_WORDS"] = df_final["TEXT"].str.split().str.len()

    mask_evaluable = df_final["HAS_TEXT"] & (df_final["TEXT_LENGTH_WORDS"] > 2)
    df_final["LANG"] = None
    df_final.loc[mask_evaluable, "LANG"] = df_final.loc[mask_evaluable, "TEXT"].apply(detectar_idioma)

    print("Posts con impacto de mercado, por plataforma:")
    print(df_final.groupby("PLATFORM")["MARKET_IMPACT"].agg(["sum", "count", "mean"]).round(3))

    # 4.10 Guardar localmente y subir/actualizar en Drive
    COLUMNAS_SALIDA = FINAL_COLUMNS + [
        "TIMESTAMP", "HOUR_MISSING", "MARKET_IMPACT", "IMPACT_CATEGORIES",
        "HAS_TEXT", "TEXT_LENGTH_CHARS", "TEXT_LENGTH_WORDS", "LANG",
    ]
    ruta_local_salida = LOCAL_DIR / OUTPUT_FILENAME
    df_final[COLUMNAS_SALIDA].to_csv(ruta_local_salida, index=False, sep=";", encoding="utf-8-sig")

    carpeta_output_id = resolver_carpeta_drive(drive_service, DRIVE_OUTPUT_PATH)
    subir_o_actualizar_archivo(drive_service, str(ruta_local_salida), carpeta_output_id, OUTPUT_FILENAME)
    print(f"Subido/actualizado en Drive: {'/'.join(DRIVE_OUTPUT_PATH)}/{OUTPUT_FILENAME}")
    print(f"Filas: {len(df_final)} | Columnas: {list(df_final[COLUMNAS_SALIDA].columns)}")
    print("Limpieza y transformación completada.")


if __name__ == "__main__":
    main()
