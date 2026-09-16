"""
Análisis semántico de las comunicaciones (TFM - sección 8.2)
Derivado de 04__Análisis_semántico.ipynb (apartados 4 y 5 del índice)

Aplica sentimiento (zero-shot + fine-tuned), NER y embeddings sobre el corpus
de comunicaciones. Diseñado para producción real:
  - El fine-tuning del modelo se hace UNA SOLA VEZ; si ya existe un modelo
    guardado en Drive, se reutiliza en vez de reentrenar.
  - El corpus se procesa de forma INCREMENTAL: solo se analizan las filas
    (por ID) que todavía no están en el fichero de salida, para que si algún
    día hay comunicaciones nuevas, no haga falta reprocesar todo el corpus.

Pensado para ejecutarse sin intervención humana (GitHub Actions), aunque hoy
se lance solo una vez a mano, porque el corpus de texto de este TFM es fijo.
"""

import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import emoji
import ftfy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline as hf_pipeline,
)
from torch.utils.data import Dataset

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

LOCAL_DIR = Path("data_semantico")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_INPUT_PATH = ["TFM DATA SCIENCE", "03. data", "04. PROCESSED - Datos Textuales"]
INPUT_FILENAME = "dataset_unificado.csv"

DRIVE_LABELED_PATH = ["TFM DATA SCIENCE", "03. data", "03. LABELS - Etiquetado de sentimiento"]
LABELED_FILENAME = "TFM_etiquetado_manual_sentimiento.xlsx"

DRIVE_OUTPUT_PATH = ["TFM DATA SCIENCE", "03. data", "05. PROCESSED - Analisis Semantico"]
OUTPUT_FILENAME = "dataset_semantico.csv"
EMBEDDINGS_FILENAME = "text_embeddings.npy"
EMBEDDINGS_INDEX_FILENAME = "text_embeddings_index.csv"

DRIVE_MODEL_PATH = ["TFM DATA SCIENCE", "03. data", "06. MODELS - Analisis Semantico"]
MODEL_ZIP_FILENAME = "modelo_sentimiento_finetuned.zip"

MODELO_BASE_FINETUNE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Columnas del corpus de entrada (heredadas del capítulo 3)
COL_ID = "ID"
COL_TEXTO = "TEXT"
COL_TIMESTAMP = "TIMESTAMP"
COL_IMPACT_CATEGORIES = "IMPACT_CATEGORIES"
COL_HAS_TEXT = "HAS_TEXT"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


# --------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y UTILIDADES DE DRIVE (mismo patrón que el módulo financiero)
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
            print(f"  Carpeta creada: {nombre}")
        else:
            raise FileNotFoundError(f"No se encontró la carpeta '{nombre}' (ruta: {' / '.join(partes_ruta[:i + 1])}).")
    return parent_id


def buscar_archivo(drive_service, carpeta_id: str, nombre_archivo: str):
    """Devuelve (file_id, mimeType) del archivo si existe en esa carpeta, o (None, None)."""
    query = f"name = '{nombre_archivo}' and '{carpeta_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    encontrados = resultado.get("files", [])
    if not encontrados:
        return None, None
    return encontrados[0]["id"], encontrados[0]["mimeType"]


def descargar_archivo(drive_service, file_id: str, mime_type: str, destino: Path):
    """Descarga un archivo de Drive. Si es un tipo nativo de Google (Sheets/Docs),
    lo exporta al formato equivalente en vez de intentar descargarlo tal cual."""
    if mime_type == "application/vnd.google-apps.spreadsheet":
        request = drive_service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        request = drive_service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    destino.write_bytes(buffer.getvalue())


def subir_o_actualizar_archivo(drive_service, ruta_local: str, carpeta_drive_id: str, nombre_archivo: str = None) -> str:
    nombre_archivo = nombre_archivo or Path(ruta_local).name
    query = f"name = '{nombre_archivo}' and '{carpeta_drive_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
    existentes = resultado.get("files", [])
    media = MediaFileUpload(ruta_local, resumable=True)
    if existentes:
        file_id = existentes[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        return file_id
    metadata = {"name": nombre_archivo, "parents": [carpeta_drive_id]}
    archivo = drive_service.files().create(body=metadata, media_body=media, fields="id").execute()
    return archivo["id"]


# --------------------------------------------------------------------------
# 3. PREPROCESAMIENTO DE TEXTO (apartado 3 del notebook, portado tal cual)
# --------------------------------------------------------------------------

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')


def limpiar_texto_base(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = ftfy.fix_text(str(texto))
    texto = HTML_TAG_PATTERN.sub(' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def preprocesar_para_transformer(texto: str) -> str:
    texto = MENTION_PATTERN.sub('@user', texto)
    return URL_PATTERN.sub('http', texto).strip()


def preparar_texto_modelo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text_clean'] = df[COL_TEXTO].apply(limpiar_texto_base)
    df['text_clean'] = df['text_clean'].apply(lambda t: emoji.demojize(t, language='es'))
    df['text_model'] = df['text_clean'].apply(preprocesar_para_transformer)
    return df


# --------------------------------------------------------------------------
# 4. SENTIMIENTO ZERO-SHOT (apartado 4.1-4.4)
# --------------------------------------------------------------------------

MAPEO_ETIQUETAS = {
    "finbert": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    "finbert_tone": {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"},
    "twitter_roberta": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    "cryptobert": {"Bullish": "positive", "Bearish": "negative", "Neutral": "neutral"},
}


def normalizar_etiqueta(modelo, etiqueta):
    return MAPEO_ETIQUETAS[modelo].get(etiqueta, etiqueta.lower())


def cargar_pipelines_zero_shot(device):
    return {
        "finbert": hf_pipeline("text-classification", model="ProsusAI/finbert", device=device),
        "finbert_tone": hf_pipeline("text-classification", model="yiyanghkust/finbert-tone", device=device),
        "twitter_roberta": hf_pipeline(
            "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device
        ),
        "cryptobert": hf_pipeline("text-classification", model="ElKulako/cryptobert", device=device),
    }


def aplicar_modelo_sentimiento(textos, pipe, nombre_modelo, batch_size=32):
    etiquetas, scores = [], []
    for i in range(0, len(textos), batch_size):
        lote = textos[i:i + batch_size]
        for r in pipe(lote, batch_size=batch_size, truncation=True, max_length=128):
            etiquetas.append(normalizar_etiqueta(nombre_modelo, r["label"]))
            scores.append(r["score"])
        if (i // batch_size) % 20 == 0:
            print(f"    {nombre_modelo}: {min(i + batch_size, len(textos))}/{len(textos)}")
    return etiquetas, scores


def aplicar_sentimiento_zero_shot(df: pd.DataFrame, pipelines: dict) -> pd.DataFrame:
    df = df.copy()
    textos_validos = df.loc[df[COL_HAS_TEXT], 'text_model'].tolist()

    for nombre_modelo in ["finbert", "finbert_tone", "twitter_roberta"]:
        df[f'sentiment_label_{nombre_modelo}'] = np.nan
        df[f'sentiment_score_{nombre_modelo}'] = np.nan
        print(f"  Aplicando {nombre_modelo}...")
        etiquetas, scores = aplicar_modelo_sentimiento(textos_validos, pipelines[nombre_modelo], nombre_modelo)
        df.loc[df[COL_HAS_TEXT], f'sentiment_label_{nombre_modelo}'] = etiquetas
        df.loc[df[COL_HAS_TEXT], f'sentiment_score_{nombre_modelo}'] = scores

    df['sentiment_label_cryptobert'] = np.nan
    df['sentiment_score_cryptobert'] = np.nan
    mask_cripto = df[COL_IMPACT_CATEGORIES].fillna("").str.contains("CRIPTO")
    if mask_cripto.sum() > 0:
        print(f"  Aplicando cryptobert ({mask_cripto.sum()} filas cripto)...")
        textos_cripto = df.loc[mask_cripto, 'text_model'].fillna("").tolist()
        etiquetas_cb, scores_cb = aplicar_modelo_sentimiento(textos_cripto, pipelines["cryptobert"], "cryptobert")
        df.loc[mask_cripto, 'sentiment_label_cryptobert'] = etiquetas_cb
        df.loc[mask_cripto, 'sentiment_score_cryptobert'] = scores_cb

    return df


# --------------------------------------------------------------------------
# 5. FINE-TUNING Y SU PERSISTENCIA EN DRIVE (apartados 4.6-4.12, adaptado)
# --------------------------------------------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_length=128):
        self.encodings = tokenizer(list(textos), truncation=True, padding=True, max_length=max_length)
        self.labels = [LABEL2ID[e] for e in etiquetas]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def cargar_dataset_etiquetado(ruta_xlsx: Path) -> pd.DataFrame:
    """Combina las dos pestañas del etiquetado manual congelado en Excel
    (equivalente fijo a las pestañas 'Etiquetado' y 'Revision_asistida' de
    la hoja de Google Sheets que se usó durante el desarrollo)."""
    df_269 = pd.read_excel(ruta_xlsx, sheet_name="Etiquetado")
    df_269 = df_269[df_269["SENTIMENT_MANUAL"].notna() & (df_269["SENTIMENT_MANUAL"].astype(str).str.strip() != "")]
    df_269 = df_269[[COL_TEXTO, "SENTIMENT_MANUAL"]].rename(columns={"SENTIMENT_MANUAL": "label"})
    df_269["origen"] = "ciego"

    df_210 = pd.read_excel(ruta_xlsx, sheet_name="Revision_asistida")
    df_210 = df_210[[COL_TEXTO, "SENTIMENT_CONFIRMADO"]].rename(columns={"SENTIMENT_CONFIRMADO": "label"})
    df_210["origen"] = "asistido"

    df_labeled = pd.concat([df_269, df_210], ignore_index=True).drop_duplicates(subset=[COL_TEXTO])
    print(f"  Dataset de etiquetado combinado: {len(df_labeled)} filas")
    print(df_labeled["label"].value_counts().to_string())
    return df_labeled


def entrenar_modelo_finetuned(df_labeled: pd.DataFrame):
    train_df, test_df = train_test_split(
        df_labeled, test_size=0.15, stratify=df_labeled["label"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE_FINETUNE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE_FINETUNE, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = SentimentDataset(train_df[COL_TEXTO].tolist(), train_df["label"].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir=str(LOCAL_DIR / "finetune_output"),
        num_train_epochs=4,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=modelo, args=training_args, train_dataset=train_dataset)
    print("  Entrenando (fine-tuning)...")
    trainer.train()
    print("  Fine-tuning completado.")
    return modelo, tokenizer


def obtener_modelo_finetuned(drive_service):
    """Si ya hay un modelo guardado en Drive, lo descarga y lo carga (sin
    reentrenar). Si no existe, entrena uno nuevo con el etiquetado manual y
    lo sube a Drive para que las próximas ejecuciones lo reutilicen."""
    carpeta_modelos_id = resolver_carpeta_drive(drive_service, DRIVE_MODEL_PATH)
    file_id, _ = buscar_archivo(drive_service, carpeta_modelos_id, MODEL_ZIP_FILENAME)

    modelo_dir = LOCAL_DIR / "modelo_finetuned"

    if file_id:
        print("Modelo fine-tuned ya existe en Drive — descargando (sin reentrenar)...")
        zip_local = LOCAL_DIR / MODEL_ZIP_FILENAME
        descargar_archivo(drive_service, file_id, "application/zip", zip_local)
        with zipfile.ZipFile(zip_local) as zf:
            zf.extractall(modelo_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(modelo_dir))
        modelo = AutoModelForSequenceClassification.from_pretrained(str(modelo_dir))
        print("Modelo cargado desde Drive.")
        return modelo, tokenizer

    print("No hay modelo guardado en Drive — entrenando por primera vez (esto solo pasa una vez).")
    carpeta_etiquetado_id = resolver_carpeta_drive(drive_service, DRIVE_LABELED_PATH)
    labeled_file_id, labeled_mime = buscar_archivo(drive_service, carpeta_etiquetado_id, LABELED_FILENAME)
    if not labeled_file_id:
        raise FileNotFoundError(f"No se encontró '{LABELED_FILENAME}' en Drive.")
    ruta_labeled = LOCAL_DIR / LABELED_FILENAME
    descargar_archivo(drive_service, labeled_file_id, labeled_mime, ruta_labeled)

    df_labeled = cargar_dataset_etiquetado(ruta_labeled)
    modelo, tokenizer = entrenar_modelo_finetuned(df_labeled)

    modelo_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(modelo_dir))
    tokenizer.save_pretrained(str(modelo_dir))

    zip_local = LOCAL_DIR / MODEL_ZIP_FILENAME
    with zipfile.ZipFile(zip_local, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in modelo_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(modelo_dir))
    subir_o_actualizar_archivo(drive_service, str(zip_local), carpeta_modelos_id, MODEL_ZIP_FILENAME)
    print("Modelo fine-tuned guardado en Drive para reutilizar en próximas ejecuciones.")
    return modelo, tokenizer


def aplicar_modelo_finetuned(df: pd.DataFrame, modelo, tokenizer, device) -> pd.DataFrame:
    df = df.copy()
    modelo.eval()
    modelo = modelo.to("cuda" if device == 0 else "cpu")

    textos = df.loc[df[COL_HAS_TEXT], "text_model"].fillna("").tolist()
    etiquetas, p_neg, p_neu, p_pos = [], [], [], []
    batch_size = 32
    for i in range(0, len(textos), batch_size):
        lote = textos[i:i + batch_size]
        inputs = tokenizer(lote, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(modelo(**inputs).logits, dim=-1).cpu().numpy()
        for p in probs:
            etiquetas.append(ID2LABEL[int(p.argmax())])
            p_neg.append(float(p[LABEL2ID["negative"]]))
            p_neu.append(float(p[LABEL2ID["neutral"]]))
            p_pos.append(float(p[LABEL2ID["positive"]]))
        if (i // batch_size) % 20 == 0:
            print(f"    fine-tuned: {min(i + batch_size, len(textos))}/{len(textos)}")

    df["sentiment_label_finetuned"] = np.nan
    df["prob_negative"] = np.nan
    df["prob_neutral"] = np.nan
    df["prob_positive"] = np.nan
    df.loc[df[COL_HAS_TEXT], "sentiment_label_finetuned"] = etiquetas
    df.loc[df[COL_HAS_TEXT], "prob_negative"] = p_neg
    df.loc[df[COL_HAS_TEXT], "prob_neutral"] = p_neu
    df.loc[df[COL_HAS_TEXT], "prob_positive"] = p_pos
    return df


def generar_embeddings(df: pd.DataFrame, modelo, tokenizer, device) -> tuple[np.ndarray, pd.Index]:
    modelo.eval()
    modelo = modelo.to("cuda" if device == 0 else "cpu")

    textos = df.loc[df[COL_HAS_TEXT], "text_model"].fillna("").tolist()
    embeddings = []
    batch_size = 32
    for i in range(0, len(textos), batch_size):
        lote = textos[i:i + batch_size]
        inputs = tokenizer(lote, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = modelo.roberta(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)
        if (i // batch_size) % 20 == 0:
            print(f"    embeddings: {min(i + batch_size, len(textos))}/{len(textos)}")

    return np.array(embeddings), df[df[COL_HAS_TEXT]].index


# --------------------------------------------------------------------------
# 6. NER Y ENTIDADES (apartado 5.1)
# --------------------------------------------------------------------------

ESLOGANES_A_EXCLUIR = {
    "american energy", "america first", "america", "energy", "enforcement",
    "military", "administration", "committee", "united states", "u. s.",
    "u. s", "country", "white house",
}

ACTIVOS_BUSCAR = {
    "S&P 500": r"\bS&P\s?500\b", "Nasdaq": r"\bnasdaq\b", "Tesla": r"\btesla\b",
    "Bitcoin": r"\bbitcoin\b|\bBTC\b", "Ethereum": r"\bethereum\b|\bETH\b",
    "Gold": r"\bgold\b", "VIX": r"\bVIX\b",
}


def limpiar_entidades(lista):
    limpio = [e.strip() for e in lista if len(e.strip()) > 2]
    return [e for e in limpio if e.lower() not in ESLOGANES_A_EXCLUIR]


def buscar_activos_mencionados(texto):
    if pd.isna(texto):
        return []
    return [nombre for nombre, patron in ACTIVOS_BUSCAR.items() if re.search(patron, texto, re.IGNORECASE)]


def aplicar_ner(df: pd.DataFrame, device) -> pd.DataFrame:
    df = df.copy()
    ner = hf_pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device)

    textos = df.loc[df[COL_HAS_TEXT], "text_model"].fillna("").tolist()
    resultados = []
    batch_size = 32
    for i in range(0, len(textos), batch_size):
        lote = textos[i:i + batch_size]
        for salida in ner(lote, batch_size=batch_size):
            resultados.append([(e["word"], e["entity_group"]) for e in salida if e["score"] > 0.5])
        if (i // batch_size) % 20 == 0:
            print(f"    NER: {min(i + batch_size, len(textos))}/{len(textos)}")

    df["ENTITIES"] = None
    df.loc[df[COL_HAS_TEXT], "ENTITIES"] = pd.Series(resultados, index=df[df[COL_HAS_TEXT]].index)

    for tipo, codigo in [("ORG", "ORG"), ("LOC", "LOC"), ("PER", "PER"), ("MISC", "MISC")]:
        col = f"ENTITIES_{tipo}"
        df[col] = df["ENTITIES"].apply(lambda ents: [e[0] for e in ents if e[1] == codigo] if ents else [])
        df[col] = df[col].apply(limpiar_entidades)

    df["ASSETS_MENTIONED"] = df["text_model"].apply(buscar_activos_mencionados)
    return df


# --------------------------------------------------------------------------
# 7. PIPELINE PRINCIPAL
# --------------------------------------------------------------------------

COLUMNAS_FINALES = [
    COL_ID, COL_TIMESTAMP, "PLATFORM", "AUTHOR", COL_TEXTO,
    "MARKET_IMPACT", COL_IMPACT_CATEGORIES, COL_HAS_TEXT,
    "sentiment_label_finbert", "sentiment_score_finbert",
    "sentiment_label_finbert_tone", "sentiment_score_finbert_tone",
    "sentiment_label_twitter_roberta", "sentiment_score_twitter_roberta",
    "sentiment_label_cryptobert", "sentiment_score_cryptobert",
    "sentiment_label_finetuned", "prob_negative", "prob_neutral", "prob_positive",
    "ENTITIES_ORG", "ENTITIES_LOC", "ENTITIES_PER", "ENTITIES_MISC", "ASSETS_MENTIONED",
]


def main():
    drive_service = get_drive_service()

    # 7.1 Descargar el corpus completo
    carpeta_input_id = resolver_carpeta_drive(drive_service, DRIVE_INPUT_PATH)
    input_file_id, input_mime = buscar_archivo(drive_service, carpeta_input_id, INPUT_FILENAME)
    if not input_file_id:
        raise FileNotFoundError(f"No se encontró '{INPUT_FILENAME}' en Drive.")
    ruta_input = LOCAL_DIR / INPUT_FILENAME
    descargar_archivo(drive_service, input_file_id, input_mime, ruta_input)

    df = pd.read_csv(ruta_input, sep=';', encoding='utf-8-sig', dtype={COL_ID: str})
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], format='mixed', errors='coerce')
    print(f"Corpus descargado: {len(df)} filas totales.")

    # 7.2 Modo incremental: descartar lo que ya está procesado en la salida
    carpeta_output_id = resolver_carpeta_drive(drive_service, DRIVE_OUTPUT_PATH)
    output_file_id, output_mime = buscar_archivo(drive_service, carpeta_output_id, OUTPUT_FILENAME)

    df_existente = None
    if output_file_id:
        ruta_existente = LOCAL_DIR / f"existente_{OUTPUT_FILENAME}"
        descargar_archivo(drive_service, output_file_id, output_mime, ruta_existente)
        df_existente = pd.read_csv(ruta_existente, sep=';', encoding='utf-8-sig', dtype={COL_ID: str})
        ids_ya_procesados = set(df_existente[COL_ID])
        df_nuevas = df[~df[COL_ID].isin(ids_ya_procesados)].copy()
        print(f"Ya había {len(df_existente)} filas procesadas — {len(df_nuevas)} filas nuevas por procesar.")
    else:
        df_nuevas = df.copy()
        print("No hay salida previa — se procesa el corpus completo (primera ejecución).")

    if df_nuevas.empty:
        print("No hay filas nuevas que procesar. Nada que hacer.")
        return

    # 7.3 Preprocesamiento
    df_nuevas = preparar_texto_modelo(df_nuevas)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Dispositivo: {'GPU' if device == 0 else 'CPU'}")

    # 7.4 Sentimiento zero-shot (4 modelos)
    print("Cargando modelos zero-shot...")
    pipelines = cargar_pipelines_zero_shot(device)
    df_nuevas = aplicar_sentimiento_zero_shot(df_nuevas, pipelines)
    del pipelines  # liberar memoria antes de cargar el modelo fine-tuned

    # 7.5 Modelo fine-tuned (entrena solo si no existe ya en Drive)
    modelo_ft, tokenizer_ft = obtener_modelo_finetuned(drive_service)
    df_nuevas = aplicar_modelo_finetuned(df_nuevas, modelo_ft, tokenizer_ft, device)

    # 7.6 NER + activos mencionados
    print("Aplicando NER...")
    df_nuevas = aplicar_ner(df_nuevas, device)

    # 7.7 Embeddings (reutilizando el modelo fine-tuned)
    print("Generando embeddings...")
    embeddings_nuevos, indices_nuevos = generar_embeddings(df_nuevas, modelo_ft, tokenizer_ft, device)

    # 7.8 Combinar con lo ya existente y guardar
    for col in ["ENTITIES_ORG", "ENTITIES_LOC", "ENTITIES_PER", "ENTITIES_MISC", "ASSETS_MENTIONED"]:
        df_nuevas[col] = df_nuevas[col].apply(lambda x: "|".join(x) if isinstance(x, list) else "")

    df_final_nuevas = df_nuevas.reindex(columns=COLUMNAS_FINALES)
    df_final = pd.concat([df_existente, df_final_nuevas], ignore_index=True) if df_existente is not None else df_final_nuevas

    ruta_salida = LOCAL_DIR / OUTPUT_FILENAME
    df_final.to_csv(ruta_salida, index=False, sep=";", encoding="utf-8-sig")
    subir_o_actualizar_archivo(drive_service, str(ruta_salida), carpeta_output_id, OUTPUT_FILENAME)
    print(f"Guardado: {OUTPUT_FILENAME} ({len(df_final)} filas totales, {len(df_final_nuevas)} nuevas)")

    # Embeddings: si ya había, se combinan; si no, se crean
    ruta_emb = LOCAL_DIR / EMBEDDINGS_FILENAME
    ruta_emb_idx = LOCAL_DIR / EMBEDDINGS_INDEX_FILENAME
    emb_file_id, emb_mime = buscar_archivo(drive_service, carpeta_output_id, EMBEDDINGS_FILENAME)
    idx_file_id, idx_mime = buscar_archivo(drive_service, carpeta_output_id, EMBEDDINGS_INDEX_FILENAME)

    if emb_file_id and idx_file_id:
        descargar_archivo(drive_service, emb_file_id, emb_mime, ruta_emb)
        descargar_archivo(drive_service, idx_file_id, idx_mime, ruta_emb_idx)
        emb_existente = np.load(ruta_emb)
        idx_existente = pd.read_csv(ruta_emb_idx)["0"].tolist()
        embeddings_final = np.vstack([emb_existente, embeddings_nuevos])
        indices_final = idx_existente + list(indices_nuevos)
    else:
        embeddings_final = embeddings_nuevos
        indices_final = list(indices_nuevos)

    np.save(ruta_emb, embeddings_final)
    pd.Series(indices_final).to_csv(ruta_emb_idx, index=False)
    subir_o_actualizar_archivo(drive_service, str(ruta_emb), carpeta_output_id, EMBEDDINGS_FILENAME)
    subir_o_actualizar_archivo(drive_service, str(ruta_emb_idx), carpeta_output_id, EMBEDDINGS_INDEX_FILENAME)
    print(f"Guardado: {EMBEDDINGS_FILENAME} ({embeddings_final.shape[0]} embeddings totales)")

    print("Análisis semántico completado.")


if __name__ == "__main__":
    main()
