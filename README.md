# TFM-PRODUCCION

Este repositorio recoge la puesta en producción del Trabajo Fin de Máster *Predicción de movimientos de mercado a partir del análisis semántico de comunicaciones públicas*, del Máster en Data Science, Big Data y Business Analytics (Universidad Complutense de Madrid).

El trabajo desarrollado en los capítulos 3 a 6 de la memoria se elaboró originalmente mediante notebooks de Google Colab, que requerían ejecución manual cada vez que se quería obtener un resultado actualizado. El objetivo de este repositorio es trasladar esa misma lógica a un proceso automatizado, de manera que el sistema se ejecute diariamente sin intervención humana. Incluye asimismo el código del agente conversacional descrito en el apartado 8.4 de la memoria, mediante el cual pueden consultarse los resultados del modelo en lenguaje natural.

Todo el sistema se apoya en herramientas de uso gratuito (GitHub Actions y Google Drive), sin necesidad de contratar ningún servicio de pago.

## Organización del repositorio

```
src/                 — los cinco módulos que componen el proceso diario, en el orden en que se ejecutan
agente/              — código del agente conversacional del apartado 8.4
.github/workflows/   — un flujo de GitHub Actions por cada módulo de src/
```

## Descripción del proceso

Los cinco módulos se ejecutan de forma encadenada: cada uno se pone en marcha únicamente cuando el anterior ha finalizado correctamente, de modo que no se procesen datos incompletos.

1. **Ingesta financiera** (`03_obtencion_datos_financieros.py`). Descarga las cotizaciones diarias de los activos considerados en el estudio (S&P 500, Nasdaq, Tesla, Bitcoin, Ethereum, entre otros) y, para los criptoactivos, incorpora información adicional procedente de Binance, como el momento exacto del máximo y el mínimo diarios.

2. **Limpieza y transformación de datos textuales** (`03_limpieza_transformacion_datos.py`). Módulo de incorporación reciente, hasta ahora no automatizado. Integra las comunicaciones de Musk, Trump y la Reserva Federal en un formato común, corrige errores de codificación del texto, elimina duplicados y determina qué comunicaciones son susceptibles de tener repercusión en los mercados.

3. **Análisis semántico** (`04_analisis_semantico.py`). Aplica distintos modelos de análisis de sentimiento (entre ellos FinBERT y un modelo ajustado específicamente para este trabajo), extrae entidades nombradas y genera representaciones vectoriales del texto.

4. **Análisis del impacto en los mercados** (`05_analisis_impacto_mercados.py`). Corresponde al núcleo metodológico del capítulo 5: estudio de eventos, comparación de la volatilidad y el volumen antes y después de cada comunicación, análisis de correlación entre sentimiento y rentabilidad, y detección de anomalías mediante autoencoders.

5. **Modelo predictivo** (`06_modelo_predictivo.py`). Entrena el modelo final (LightGBM, seleccionado tras comparar su rendimiento con XGBoost y Random Forest) y calcula diariamente la probabilidad de que se produzca un movimiento relevante en cada activo.

Únicamente el primer módulo dispone de una programación horaria fija (diariamente a las 6:00 UTC); el resto se activa automáticamente en cuanto concluye con éxito el módulo precedente.

## El agente conversacional (`agente/`)

Se trata de la aplicación de Streamlit que permite consultar los resultados del sistema en lenguaje natural: la predicción del día, las variables que más influyeron en ella, o la interpretación de un comunicado concreto. En una versión anterior se incorporó el dashboard de Tableau directamente en la interfaz del agente; esta opción se descartó porque no se integraba adecuadamente desde el punto de vista visual y no permitía adaptar la información a la consulta planteada por el usuario. Por este motivo, el agente genera actualmente sus propias visualizaciones mediante Plotly, y el dashboard de Tableau se mantiene como un elemento independiente, sin conexión entre ambos.

## Almacenamiento de los datos

Todos los datos se conservan en una carpeta de Google Drive compartida por el equipo (`TFM DATA SCIENCE`), organizada mediante carpetas numeradas conforme al orden del proceso (01. RAW - Datos Financieros, 02. RAW - Datos Textuales, y así sucesivamente hasta 08. PROCESSED - Modelado).

El acceso se realiza mediante una cuenta de servicio de Google Cloud, cuya clave se almacena como secreto de GitHub y no figura en el código. Debe tenerse en cuenta que este tipo de cuentas carece de cuota propia de almacenamiento en Drive, por lo que únicamente puede actualizar archivos ya existentes, no crear carpetas ni archivos nuevos. En caso de modificar la ubicación de alguna carpeta de salida, es necesario crear manualmente, desde una cuenta de usuario con cuota disponible, los archivos vacíos correspondientes antes de ejecutar el módulo afectado; de lo contrario, la ejecución finaliza con un error de cuota excedida.

## Limitaciones

- El corpus de comunicaciones es cerrado: no se incorporan comunicaciones nuevas de forma automática, salvo que se ejecute manualmente el módulo de ingesta correspondiente. En consecuencia, la predicción diaria depende principalmente de la evolución de las condiciones financieras, no del componente semántico.
- El dashboard de Tableau Public no mantiene una conexión en tiempo real con el proceso automatizado: su actualización requiere descargar el CSV correspondiente y volver a publicar el documento manualmente.
- GitHub Actions no ofrece acceso a GPU en su capa gratuita, por lo que el entrenamiento e inferencia del modelo de sentimiento se ejecutan sobre CPU.
- Si el repositorio permanece sin actividad durante 60 días, GitHub desactiva automáticamente los flujos programados mediante cron.
