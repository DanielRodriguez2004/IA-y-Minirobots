hazlo mucho mas breve, que solo sea copiar y pegar en el github
# Taller 3 y Taller 5 - Inteligencia Artificial

## Taller 3 - Punto 2: Verdadera democracia

Se desarrolló un algoritmo genético para distribuir 50 entidades estatales entre 5 partidos políticos de acuerdo con su representación en un congreso de 50 curules.

Cada entidad tiene un peso político aleatorio entre 1 y 100. El algoritmo busca que el poder total asignado a cada partido sea lo más cercano posible al poder que le corresponde proporcionalmente por su número de curules.

Se utilizaron:
- Selección por torneo
- Cruce uniforme
- Mutación
- Elitismo

Archivo principal:

`Taller3_Punto2_Verdadera_Democracia.ipynb`

El notebook genera la distribución de curules, la matriz de poder, la comparación entre poder objetivo y asignado, y la gráfica de convergencia.

---

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
