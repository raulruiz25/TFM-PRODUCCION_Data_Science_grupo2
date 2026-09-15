# TFM-PRODUCCION

Arquitectura de producción del Trabajo Fin de Máster **"Predicción de movimientos de mercado a partir del análisis semántico de comunicaciones públicas"** (Máster en Data Science, Big Data y Business Analytics, UCM, curso 2025/2026).

Este repositorio contiene tanto el pipeline automatizado (capítulo 8.2 de la memoria) como el código del agente conversacional de explicabilidad (capítulo 8.4).

## Estructura del repositorio

```
TFM-PRODUCCION/
├── src/                        # Los 4 módulos de producción
│   ├── 03_obtencion_datos_financieros.py
│   ├── 04_analisis_semantico.py
│   ├── 05_analisis_impacto_mercados.py
│   └── 06_modelo_predictivo.py
├── agente/                     # Agente conversacional de explicabilidad (Streamlit)
│   ├── app.py
│   ├── carga_datos.py
│   ├── router_intencion.py
│   ├── respuestas.py
│   ├── simulacion.py
│   └── requirements.txt
├── .github/workflows/          # Un workflow por módulo de src/
└── requirements*.txt           # Dependencias separadas por módulo
```

## Qué hace este pipeline

El sistema estima diariamente la probabilidad de que ocurra un movimiento de mercado atípico en un conjunto de activos financieros (S&P 500, Nasdaq, Tesla, ETF de Energía, Bitcoin, Ethereum), combinando:

- **Datos financieros** actualizados a diario (precios, volumen, volatilidad).
- **Comunicaciones públicas** de Donald Trump, Elon Musk y la Reserva Federal, analizadas semánticamente (sentimiento, entidades, temáticas). El corpus de comunicaciones es una colección **cerrada** (sin incorporación de contenido nuevo desde diciembre de 2025 para Musk), por lo que este componente no aporta información nueva día a día aunque el módulo correspondiente sí se ejecute a diario (ver más abajo).

Todo el stack funciona con tecnologías de coste cero: sin necesidad de vincular ninguna cuenta de facturación en ningún servicio.

## Arquitectura

- **Orquestación y cómputo:** GitHub Actions.
- **Almacenamiento:** Google Drive (repositorio compartido de datos en bruto y procesados), accedido mediante una cuenta de servicio de Google Cloud (`tfm-ingesta-financiera@...`), cuya clave se gestiona como secreto cifrado del repositorio (`GDRIVE_SERVICE_ACCOUNT_KEY`).
- **Modelos:** se serializan y guardan directamente en Google Drive (formato `joblib` para el modelo predictivo, formato nativo de `transformers` para el modelo de sentimiento ajustado).

## Módulos de producción (`src/`)

Los cuatro módulos se ejecutan **encadenados cada mañana**, uno detrás de otro, mediante disparadores `workflow_run`: cada workflow arranca solo si el anterior terminó con éxito (o si se lanza a mano con `workflow_dispatch`), de modo que ningún módulo procesa datos potencialmente incompletos.

```
Ingesta financiera (cron 06:00 UTC)
        │  workflow_run (success)
        ▼
Análisis semántico
        │  workflow_run (success)
        ▼
Análisis de impacto en mercados
        │  workflow_run (success)
        ▼
Modelo predictivo
```

| # | Script | Qué hace | Modo de cálculo |
|---|--------|----------|------------------|
| 1 | `03_obtencion_datos_financieros.py` | Descarga precios de cierre de los 11 activos vía `yfinance`, enriquecido con datos intradía de Binance para Bitcoin y Ethereum (hora del máximo/mínimo, % de volumen en órdenes de mercado). Único disparador por `cron` (06:00 UTC); el resto de módulos se disparan por encadenamiento. | Incremental (últimos 5 días, fusiona con histórico). Modo `backfill` disponible para carga completa única. |
| 2 | `04_analisis_semantico.py` | Aplica 4 modelos de sentimiento zero-shot (FinBERT, FinBERT-tone, twitter-RoBERTa, CryptoBERT) y el modelo twitter-RoBERTa ajustado sobre muestra etiquetada manualmente; extrae entidades (NER) y embeddings. **Se ejecuta automáticamente cada día** (encadenado tras el módulo 1), aunque al ser el corpus de texto cerrado, normalmente no encuentra comunicaciones nuevas que procesar y termina rápido. | Incremental por ID (solo procesa filas no vistas antes). El modelo ajustado se reutiliza desde Drive; el fine-tuning no se repite. |
| 3 | `05_analisis_impacto_mercados.py` | Estudio de eventos, comparación de volatilidad/volumen antes-después de cada comunicación, correlación sentimiento-retorno, detección de anomalías con autoencoders. Genera además el dataset consolidado que alimenta el módulo 4. | Recalcula la serie completa cada vez (no incremental): los modelos estadísticos dependen del histórico completo. |
| 4 | `06_modelo_predictivo.py` | Construye las variables predictoras, entrena el modelo final (**LightGBM**, `n_estimators=200, max_depth=5, learning_rate=0.05`, umbral de decisión 0.5), genera informes de interpretabilidad (SHAP, comparación de modelos, AUC por activo, matriz de confusión por umbral) y calcula la predicción del día para cada uno de los 6 activos con evidencia suficiente. | Reentreno completo cada día (no se cachea el modelo, a diferencia del módulo 2), para demostrar el pipeline de reentrenamiento automático de principio a fin. |

> **Nota sobre el módulo 4:** el docstring del script todavía dice "Random Forest, la elección del TFM" — es un comentario desfasado de una versión anterior; el modelo que realmente se entrena y sirve como `modelo_final` en el código es LightGBM, coherente con la memoria.

## Agente conversacional de explicabilidad (`agente/`)

Código de la app de Streamlit del capítulo 8.4, desplegada en Streamlit Community Cloud. Se estructura en:

- **`app.py`** — interfaz principal: una sola columna de chat centrada. Cada respuesta que se apoya en datos concretos lleva su propio gráfico de Plotly pegado justo debajo, dentro de la misma burbuja del historial.
- **`carga_datos.py`** — descarga desde Drive el modelo predictivo, el modelo de sentimiento ajustado y los informes de interpretabilidad al arrancar la app.
- **`router_intencion.py`** — clasifica cada mensaje del usuario en SIMULACIÓN (comunicado nuevo a analizar) o PREGUNTA_DATOS (consulta sobre resultados ya calculados). Deliberadamente basado en palabras clave, no en un LLM, para que funcione siempre sin depender de una API externa ni de su cuota.
- **`simulacion.py`** — calcula el sentimiento de un comunicado nuevo (real o hipotético) y compara la probabilidad de evento importante antes/después, con las condiciones de mercado más recientes.
- **`respuestas.py`** — redacta las respuestas finales, con la API de Gemini si está disponible o con una plantilla de respaldo construida a partir de los mismos datos.

**Cambio relevante respecto a una versión anterior:** el agente incluía en su día el dashboard de Tableau Public embebido en un panel aparte. Se sustituyó por los gráficos nativos de Plotly porque el dashboard no encajaba visualmente con el resto de la app (fondo propio, tamaños fijos, barra de herramientas ajena) y porque un panel fijo no podía adaptarse a la pregunta concreta del usuario. El dashboard de Tableau sigue publicado como pieza independiente, sin enlace funcional desde la app.

## Almacenamiento en Google Drive

Cada módulo lee y escribe en carpetas dedicadas del Drive compartido del proyecto (`TFM DATA SCIENCE`):

- `data/PROCESSED - Datos Textuales/` — corpus de comunicaciones unificado.
- `04. Análisis semántico/` — etiquetas manuales de sentimiento congeladas.
- `PROCESSED - Impacto Mercados/` — resultados del módulo 3.
- `08. PROCESSED - Modelado/` — resultados del módulo 4 (`predicciones_hoy.csv`, informes de interpretabilidad, modelo serializado).

**Nota sobre cuota de almacenamiento:** las cuentas de servicio de Google Cloud no tienen cuota propia en Drive, por lo que solo pueden **actualizar** archivos ya existentes, no crear archivos nuevos. Si se despliega un módulo que escribe en una carpeta nueva de Drive por primera vez, es necesario crear manualmente (desde una cuenta de usuario con cuota) los ficheros vacíos de destino antes de la primera ejecución automatizada.

## Requisitos para ejecutar o desplegar

- Python (ver el `requirements*.txt` correspondiente a cada módulo/carpeta).
- Credenciales de la cuenta de servicio de Google Cloud con acceso a la API de Drive, configuradas como secreto de GitHub Actions (`GDRIVE_SERVICE_ACCOUNT_KEY`).
- Acceso de lectura al Drive compartido `TFM DATA SCIENCE`.
- Para el agente: clave de la API de Gemini (opcional; si no está disponible, usa plantilla de respaldo).

## Relación con el resto del TFM

- El **dashboard de Tableau Public** (capítulo 7 de la memoria) no se conecta en vivo a este pipeline: se alimenta de CSV exportados y republicados manualmente cuando se quiere reflejar una nueva ejecución, y funciona como pieza totalmente independiente del agente (capítulo 8.6).
- El **agente conversacional de explicabilidad** (capítulo 8.4), desplegado por separado en Streamlit Community Cloud, sí consume directamente los resultados que este pipeline deja en Drive.

## Limitaciones conocidas

- El análisis semántico no incorpora comunicaciones nuevas desde diciembre de 2025 (Musk) por ser un corpus cerrado; la predicción diaria del módulo 4 está por tanto dominada por el componente financiero.
- GitHub Actions no ofrece GPU en su capa gratuita: el ajuste fino y la inferencia de los modelos de sentimiento corren sobre CPU.
- GitHub deshabilita automáticamente los workflows programados por cron tras 60 días de inactividad en el repositorio.
