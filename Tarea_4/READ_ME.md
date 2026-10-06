
## Taller 5 - Punto 3: Clasificación con dataset de Kaggle

Se utilizó el dataset **Breast Cancer Wisconsin (Diagnostic)** descargado desde Kaggle mediante `kagglehub`.

El objetivo fue clasificar muestras como:

- Benigno
- Maligno

Se entrenó una red neuronal con TensorFlow/Keras.

Resultados principales:

- Accuracy: aproximadamente 97 %
- 72 muestras benignas clasificadas correctamente
- 39 muestras malignas clasificadas correctamente
- 3 falsos negativos

Archivo principal:

`Taller5_Punto3_Kaggle_Cancer.ipynb`

---

## Taller 5 - Punto 4: Predicción del valor de viviendas

Como punto libre se desarrolló una red neuronal de regresión utilizando el dataset **California Housing**.

El modelo predice el valor medio de las viviendas a partir de características como ingreso medio, antigüedad, número de habitaciones, población y ubicación.

Resultados:

- MAE: 0.3684
- RMSE: 0.5415
- R²: 0.7762

El modelo logra explicar aproximadamente el 77.62 % de la variabilidad de los datos.

Archivo principal:

`Taller5_Punto4_California_Housing.ipynb`

---

## Ejecución

Los notebooks están preparados para ejecutarse en Google Colab.

1. Abrir Google Colab.
2. Subir el archivo `.ipynb`.
3. Ejecutar todas las celdas en orden.

Dependencias principales:

`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `kagglehub`

