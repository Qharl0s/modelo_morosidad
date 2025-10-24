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
