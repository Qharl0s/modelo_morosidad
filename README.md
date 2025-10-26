# 🧠 Modelo Predictor de Morosidad

## 📚 Random Forest y Redes Neuronales

Este proyecto desarrolla un modelo de predicción de morosidad utilizando técnicas de Machine Learning, principalmente Random Forest y Redes Neuronales, para analizar el comportamiento crediticio y estimar la probabilidad de incumplimiento.

## ⚙️ Requisitos previos

🐍 Python 3.8+.

## 💻 Crear y activar entorno virtual

    python -m venv venv

    venv\Scripts\activate


## 📦 Instalación de dependencias
    
    # para Random Forest
    pip install pandas scikit-learn matplotlib seaborn openpyxl
    
    # para Red Neuronal
    pip install pandas numpy scikit-learn tensorflow openpyxl

## 📈 Ejecución y Resultados:

🔹 Random Forest

    # Random Forest:
    python random_forest.py

   Matriz de confusión:

|               | Predicción Positiva | Predicción Negativa |
|---------------|--------------------:|--------------------:|
| Real Positivo | 3651               | 8                  |
| Real Negativo | 59                 | 2828               |

   Reporte de Clasificación

| Clase | Precisión | Recall | F1-Score | Soporte |
|:------|:----------:|:------:|:---------:|:--------:|
| 0 | 0.98 | 1.00 | 0.99 | 3659 |
| 1 | 1.00 | 0.98 | 0.99 | 2887 |
| **Exactitud Global** |  |  | **0.99** | 6546 |

🔹 Red Neuronal:

    # Red Neuronal:
    python red_neuronal.py

   Matriz de Confusión:

|               | Predicción Positiva | Predicción Negativa |
|---------------|--------------------:|--------------------:|
| Real Positivo | 3534                | 108                 |
| Real Negativo | 249                 | 2655                |


   Reporte de Clasificación

| Clase | Precisión | Recall | F1-Score | Soporte |
|:------|:----------:|:------:|:---------:|:--------:|
| 0 | 0.93 | 0.97 | 0.95 | 3642 |
| 1 | 0.96 | 0.91 | 0.94 | 2904 |
| **Exactitud Global** |  |  | **0.95** | 6546 |







