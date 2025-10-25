🧠 Modelo Predictor de Morosidad

🔍 Random Forest y Redes Neuronales

Este proyecto desarrolla un modelo de predicción de morosidad utilizando técnicas de Machine Learning, principalmente Random Forest y Redes Neuronales, para analizar el comportamiento crediticio y estimar la probabilidad de incumplimiento.

⚙️ Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado Python 3.8+.
Se recomienda trabajar dentro de un entorno virtual.

🔧 Crear y activar entorno virtual

En Windows:

python -m venv venv

venv\Scripts\activate

En macOS / Linux:

python3 -m venv venv

source venv/bin/activate


📦 Instalación de dependencias

Para ejecutar el modelo basado en Random Forest, instala las siguientes librerías:

pip install pandas scikit-learn matplotlib seaborn openpyxl

Para ejecutar el modelo basado en Redes Neuronales, instala las siguientes librerías:

pip install pandas numpy scikit-learn tensorflow openpyxl

Resultados:

Random Forest:

🔹 Matriz de confusión:
[[3651    8]
 [  59 2828]]


🔹 Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      3659
           1       1.00      0.98      0.99      2887

    accuracy                           0.99      6546
   
   macro avg       0.99      0.99      0.99      6546
   
weighted avg       0.99      0.99      0.99      6546


Red Neuronal:

🔹 Matriz de Confusión:
[[3534  108]
 [ 249 2655]]

🔹 Reporte de Clasificación:
              precision    recall  f1-score   support

           0       0.93      0.97      0.95      3642
           1       0.96      0.91      0.94      2904

    accuracy                           0.95      6546
    
   macro avg       0.95      0.94      0.94      6546
   
weighted avg       0.95      0.95      0.95      6546




