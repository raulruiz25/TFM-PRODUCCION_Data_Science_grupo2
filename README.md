# TFM-PRODUCCION

Este repositorio es la parte "en producción" de mi TFM del Máster en Data Science, Big Data y Business Analytics (UCM): *Predicción de movimientos de mercado a partir del análisis semántico de comunicaciones públicas*.

La idea es sencilla: todo el trabajo del TFM (capítulos 3 a 6) empezó como notebooks de Colab que había que ejecutar a mano cada vez. Aquí lo que hicimos fue coger esa misma lógica y automatizarla con GitHub Actions, para que el sistema completo se ejecute solo cada día sin que nadie tenga que abrir un notebook. También está el código del agente conversacional del capítulo 8.4, que es la parte con la que se puede "hablar" con los resultados del modelo.

Todo funciona con herramientas gratuitas (GitHub Actions, Google Drive), así que no hace falta pagar nada ni dar de alta ninguna tarjeta en ningún sitio.

## Cómo está organizado

```
src/                → los 5 scripts que se ejecutan cada día, en orden
agente/              → la app de Streamlit del capítulo 8.4
.github/workflows/   → un workflow de GitHub Actions por cada script de src/
```

## Qué hace cada paso, en orden

El pipeline se ejecuta como una cadena: cada paso solo arranca si el anterior ha terminado bien. Si uno falla, los siguientes no se ejecutan, para no trabajar con datos a medias.

1. **Ingesta financiera** (`03_obtencion_datos_financieros.py`) — descarga los precios diarios de los activos que usamos (S&P 500, Nasdaq, Tesla, Bitcoin, Ethereum, etc.) y para Bitcoin y Ethereum añade también algún dato extra de Binance, como a qué hora del día tocó el máximo o el mínimo.

2. **Limpieza de datos textuales** (`03_limpieza_transformacion_datos.py`) — este es nuevo, no estaba automatizado antes. Coge los tuits de Musk, los Truths de Trump y los comunicados de la Fed en bruto, los junta en un único formato, corrige problemas de codificación de texto, quita duplicados y marca qué comunicaciones podrían tener relación con los mercados.

3. **Análisis semántico** (`04_analisis_semantico.py`) — le pasa varios modelos de sentimiento al texto (FinBERT, un modelo ajustado a mano por nosotros, etc.), saca entidades nombradas (empresas, países...) y genera embeddings.

4. **Análisis de impacto en mercados** (`05_analisis_impacto_mercados.py`) — el corazón del capítulo 5: estudio de eventos, mira si hay diferencias de volatilidad y volumen antes y después de cada comunicación, correlación entre sentimiento y retorno, y detección de anomalías con autoencoders.

5. **Modelo predictivo** (`06_modelo_predictivo.py`) — entrena el modelo final (LightGBM, que fue el que mejor funcionó frente a XGBoost y Random Forest) y calcula cada día la probabilidad de que pase algo importante en cada activo.

Solo el primer paso tiene un cron fijo (todos los días a las 6:00 UTC); el resto se disparan automáticamente en cuanto el anterior termina bien.

## El agente (carpeta `agente/`)

Es la app de Streamlit desde la que se puede preguntar cosas sobre los resultados: qué predijo el modelo hoy, por qué, qué variables pesaron más, etc. Al principio la idea era meter el dashboard de Tableau dentro de la propia app, pero al final no encajaba bien visualmente y no se adaptaba a lo que preguntara cada usuario, así que ahora el agente genera sus propios gráficos con Plotly y el dashboard de Tableau se quedó como algo aparte, sin conexión entre los dos.

## Dónde se guardan los datos

Todo se guarda en una carpeta de Google Drive compartida del equipo (`TFM DATA SCIENCE`), organizada por carpetas numeradas según el orden del pipeline (01. RAW - Datos Financieros, 02. RAW - Datos Textuales, y así hasta 08. PROCESSED - Modelado).

El acceso lo hace una cuenta de servicio de Google Cloud, cuya clave está guardada como secreto en GitHub (no en el código). Una cosa a tener en cuenta: este tipo de cuentas no tiene cuota propia de almacenamiento en Drive, así que solo puede actualizar ficheros que ya existen, no crear carpetas o ficheros nuevos desde cero. Si en algún momento cambiáis de sitio alguna carpeta de salida, hay que crear a mano (desde una cuenta normal) los ficheros vacíos de destino antes de que corra el script, si no falla con un error de cuota.

## Cosas a tener en cuenta

- El corpus de comunicaciones es cerrado: no se están metiendo tuits o comunicados nuevos día a día (salvo que alguien vuelva a ejecutar la ingesta a mano). Por eso la predicción diaria depende sobre todo de cómo se mueve el mercado ese día, no tanto del texto.
- El dashboard de Tableau Public no está conectado en vivo al pipeline: cuando queremos que refleje una ejecución nueva, hay que bajar el CSV actualizado y volver a publicar el workbook a mano.
- GitHub Actions no da GPU gratis, así que el modelo de sentimiento se entrena y se usa en CPU.
- Si el repositorio está más de 60 días sin actividad, GitHub desactiva solo los workflows programados por cron.
