# ======================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ======================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ======================================
# 2. CARGA DE DATOS
# ======================================
df = pd.read_excel("data.xlsx")

# Limpieza de nombres de columnas
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ======================================
# 3. PREPROCESAMIENTO
# ======================================
cat_cols = [c for c in ['Sexo', 'Actividad', 'Producto', 'Agencia'] if c in df.columns]
num_cols = [c for c in ['Edad', 'Ingresos', 'Capital_Desembolsado',
                        'Tasa_Compensatoria', 'Cuotas', 'Cuotas_Atrasadas', 'AtrasoRCC'] if c in df.columns]

# Codificación de categóricas
if cat_cols:
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
else:
    df_encoded = pd.DataFrame()

# Escalado de numéricas
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# Unir numéricas + categóricas
X = pd.concat([df_scaled, df_encoded], axis=1)

# Variable objetivo
y = df['Moroso']
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# ======================================
# 4. DIVISIÓN DE DATOS
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# 5. CREACIÓN DEL MODELO AJUSTADO (11 variables + 1 resultado)
# ======================================

model = Sequential([
    # Dense(128, input_dim=11, activation='relu'),   # Capa de entrada (11 variables) + 1ª capa oculta
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),                                  # Apaga el 30% de nodos durante entrenameinto
    Dense(64, activation='relu'),                  # 2ª capa oculta
    Dropout(0.2),                                  # Apaga el 20% de nodos durante entrenameinto
    Dense(32, activation='relu'),                  # 3ª capa oculta
    Dense(1, activation='sigmoid')                 # Capa de salida binaria (moroso / no moroso)
])

model.compile(
    optimizer='adam',                  # Optimizador adaptativo
    loss='binary_crossentropy',        # Función de pérdida binaria
    metrics=['accuracy']               # Métrica principal: precisión
)


# ======================================
# 6. ENTRENAMIENTO
# ======================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ======================================
# 7. EVALUACIÓN
# ======================================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🔹 Precisión en test: {acc:.4f}")

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("\n🔹 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n🔹 Matriz de Confusión:")
print(cm)

# ======================================
# 8. VISUALIZACIONES DE RENDIMIENTO
# ======================================

## 8.1 Matriz de Confusión Normalizada
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8,6))
sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["No Moroso", "Moroso"],
            yticklabels=["No Moroso", "Moroso"])
plt.title("Matriz de Confusión Normalizada (%)")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

## 8.2 Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", lw=2)
plt.plot([0,1], [0,1], 'k--')
plt.title("Curva ROC")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.legend(loc="lower right")
plt.show()

## 8.3 Curva Precisión-Recall
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, label=f'AP = {ap:.3f}', lw=2)
plt.title("Curva Precisión-Recall")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.legend(loc="best")
plt.show()

## 8.4 Distribución de probabilidades predichas
plt.figure(figsize=(8,6))
sns.histplot(y_prob[y_test==0], color='green', label='No Moroso', kde=True, stat="density")
sns.histplot(y_prob[y_test==1], color='red', label='Moroso', kde=True, stat="density")
plt.title("Distribución de Probabilidades Predichas")
plt.xlabel("Probabilidad de ser Moroso")
plt.legend()
plt.show()

## 8.5 Evolución de Precisión
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Evolución de la Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

## 8.6 Evolución de la Pérdida
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# # ======================================
# # 9. GUARDADO DEL MODELO
# # ======================================
# model.save("modelo_morosidad.h5")
# print("\n✅ Modelo guardado como 'modelo_morosidad.h5'")
