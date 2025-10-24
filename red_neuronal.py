# ==============================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 2. CARGA DE DATOS DESDE EXCEL
# ==============================
df = pd.read_excel("data.xlsx")

# Limpieza de nombres de columnas (quita espacios y acentos)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")



# ==============================
# 3. PREPROCESAMIENTO
# ==============================

# Ajusta estos nombres si tus columnas tienen variaciones
cat_cols = [c for c in ['Sexo', 'Actividad', 'Producto', 'Agencia'] if c in df.columns]
num_cols = [c for c in ['Edad', 'Ingresos', 'Capital_Desembolsado', 
                        'Tasa_Compensatoria', 'Cuotas', 'Cuotas_Atrasadas', 'AtrasoRCC'] if c in df.columns]

# Codificación de variables categóricas
if cat_cols:
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
else:
    df_encoded = pd.DataFrame()

# Escalado de variables numéricas
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# Concatenamos todo
X = pd.concat([df_scaled, df_encoded], axis=1)

# Variable objetivo
y = df['Moroso']

# Si no es numérica, la convertimos
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Convertimos a categorías (para clasificación multiclase)
y_cat = to_categorical(y)

# ==============================
# 4. DIVISIÓN ENTRE TRAIN Y TEST
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)


# ==============================
# 5. CREACIÓN DEL MODELO
# ==============================
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 6. ENTRENAMIENTO
# ==============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ==============================
# 7. EVALUACIÓN
# ==============================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🔹 Precisión en test: {acc:.4f}")