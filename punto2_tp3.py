import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# 1. IMPORTACIÓN DE LAS IMÁGENES
# Descargamos las partes que nos da Keras
(X_train_keras, y_train_keras), (X_test_keras, y_test_keras) = mnist.load_data()

# Unimos todo para tener el dataset original completo de 70.000 muestras
X_full = np.concatenate((X_train_keras, X_test_keras))
y_full = np.concatenate((y_train_keras, y_test_keras))

# 2. APLANAMIENTO
# Pasamos de tener 70.000 matrices de (28x28) a 70.000 vectores unidimensionales de 784 posiciones
X_flat = X_full.reshape(X_full.shape[0], 784)

# 3. NORMALIZACIÓN
# Transformamos la escala de píxeles enteros (0 a 255) a flotantes (0.0 a 1.0)
X_norm = X_flat / 255.0

# 4. SEPARACIÓN 80/20
# test_size=0.20 garantiza que exactamente el 20% se guarde para testing
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_full, test_size=0.20, random_state=42
)

# Imprimimos para confirmar que todo quedó con las dimensiones correctas
print(f"Total de imágenes procesadas: {X_norm.shape[0]}")
print(f"Imágenes para Entrenamiento (80%): {X_train.shape[0]} muestras de {X_train.shape[1]} píxeles")
print(f"Imágenes para Evaluación (20%):  {X_test.shape[0]} muestras de {X_test.shape[1]} píxeles")