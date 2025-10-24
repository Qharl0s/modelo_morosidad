import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


# Cargar archivo
df = pd.read_excel("data.xlsx")

# Filtrar clases válidas (0 a 4)
df = df[df["Moroso"].isin([0, 1, 2, 3, 4])]

# Codificar variables categóricas
label_encoders = {}
for col in df.select_dtypes(include="object"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar variables predictoras y objetivo
X = df.drop("Moroso", axis=1)
y = df["Moroso"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest con balanceo
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title("Matriz de Confusión")
plt.show()

# Importancia de variables
importances = model.feature_importances_
feature_names = X.columns
feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Importancia de las variables")
plt.tight_layout()
plt.show()

# Visualizar uno de los árboles del bosque
plt.figure(figsize=(20, 10))  # Ajusta el tamaño según necesidad
plot_tree(
    model.estimators_[0],                # Uno de los árboles del Random Forest
    feature_names=X.columns,             # Nombres de las variables
    class_names=[str(cls) for cls in model.classes_],  # Nombres de clases
    filled=True,                         # Colores según clase
    rounded=True,                        # Bordes redondeados
    fontsize=6                          # Tamaño de letra
)
plt.title("Árbol de decisión del Random Forest (estimador 0)")
plt.show()

# Histograma de Cuotas Atradas vs Moroso
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='nEdad', hue='Moroso', multiple="stack", bins=20)
plt.title("Cuotas atrasadas según nivel de morosidad")
plt.xlabel("Cuotas atrasadas")
plt.ylabel("Frecuencia")
plt.show()

# Matriz de Dispersion
# Filtrar solo morosos de nivel 3 y 4
df_filtrado = df[df['Moroso'].isin([3, 4])]


if 'Producto' in label_encoders:
    df_filtrado['Producto'] = label_encoders['Producto'].inverse_transform(df_filtrado['Producto'])

# Gráfico de dispersión
plt.figure(figsize=(12, 6))
sns.stripplot(
    data=df_filtrado,
    x='Producto',           # Variable categórica en eje X
    y='Capital_Desembolsado',        # Variable numérica en eje Y
    hue='Moroso',          # Color según nivel de morosidad
    jitter=True,            # Desplaza puntos horizontalmente para que no se encimen
    dodge=True,
    palette='coolwarm',
    alpha=0.7
)
plt.title("Producto x Desembolso (Morosos Nivel 3 y 4)")
plt.xlabel("Producto")
plt.ylabel("Desembolso")
plt.legend(title="Moroso")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()