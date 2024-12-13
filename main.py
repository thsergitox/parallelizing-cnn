import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
from joblib import Parallel, delayed
import networkx as nx
import time

# --------------------------------------------------------------------------------------------
# 1. CARGA DEL DATASET MNIST
# --------------------------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# --------------------------------------------------------------------------------------------
# 2. PREPROCESAMIENTO
# --------------------------------------------------------------------------------------------
# Se transforman las imágenes de 28x28 a un vector de 784 (28*28)
X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test  = X_test.reshape(-1, 28 * 28) / 255.0

# --------------------------------------------------------------------------------------------
# 3. DIVISIÓN DEL CONJUNTO DE ENTRENAMIENTO EN SUBCONJUNTOS
# --------------------------------------------------------------------------------------------
subsets = 4  # Número de subconjuntos, y por ende, número de modelos en paralelo
X_train_splits = np.array_split(X_train, subsets)
y_train_splits = np.array_split(y_train, subsets)

# --------------------------------------------------------------------------------------------
# 4. FUNCIÓN PARA ENTRENAR UN SUBCONJUNTO DE DATOS
# --------------------------------------------------------------------------------------------
def train_subset(X_subset, y_subset):
    """
    Crea y entrena un MLPClassifier simple (1 capa oculta de tamaño 128).
    Entrena sólo con un subconjunto de datos (X_subset, y_subset).
    """
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42, verbose=False)
    model.fit(X_subset, y_subset)
    return model

# --------------------------------------------------------------------------------------------
# 5. ENTRENAMIENTO EN PARALELO
# --------------------------------------------------------------------------------------------
start_parallel = time.time()
parallel_models = Parallel(n_jobs=subsets)(
    delayed(train_subset)(X_part, y_part) for X_part, y_part in zip(X_train_splits, y_train_splits)
)
end_parallel = time.time()

# --------------------------------------------------------------------------------------------
# 6. FUNCIÓN DE ENSEMBLE PARA VOTACIÓN MAYORITARIA
# --------------------------------------------------------------------------------------------
def ensemble_predict(models, X):
    """
    Combina las predicciones de cada modelo usando una votación mayoritaria.
    Este método NO concatena pesos ni capas; sólo combina predicciones.
    """
    predictions = np.array([model.predict(X) for model in models])
    # predictions.shape = (num_modelos, num_muestras)
    final_predictions = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
    )
    return final_predictions

ensemble_predictions = ensemble_predict(parallel_models, X_test)

# --------------------------------------------------------------------------------------------
# 7. EVALUACIÓN DEL MODELO EN PARALELO
# --------------------------------------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, ensemble_predictions)
report = classification_report(y_test, ensemble_predictions)

# --------------------------------------------------------------------------------------------
# 8. ENTRENAMIENTO SECUENCIAL (BASELINE)
# --------------------------------------------------------------------------------------------
start_sequential = time.time()
seq_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42, verbose=True)
seq_model.fit(X_train, y_train)
end_sequential = time.time()

seq_predictions = seq_model.predict(X_test)
seq_conf_matrix = confusion_matrix(y_test, seq_predictions)
seq_report = classification_report(y_test, seq_predictions)

# --------------------------------------------------------------------------------------------
# 9. GRÁFICAS: MATRICES DE CONFUSIÓN
# --------------------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Parallel Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(seq_conf_matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Sequential Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# --------------------------------------------------------------------------------------------
# 10. GRÁFICO DE DEPENDENCIAS (EJEMPLO CON NETWORKX)
# --------------------------------------------------------------------------------------------
G = nx.DiGraph()
G.add_edges_from([
    ("Input", "Hidden Layer"),
    ("Hidden Layer", "Output")
])
plt.figure(figsize=(6, 6))
nx.draw_networkx(G, with_labels=True, node_color='skyblue', node_size=3000, font_size=15)
plt.title("Dependency Graph")
plt.show()

# --------------------------------------------------------------------------------------------
# 11. IMPRESIÓN DE RESULTADOS
# --------------------------------------------------------------------------------------------
print("Classification Report - Parallel Model:\n", report)
print("Classification Report - Sequential Model:\n", seq_report)
print(f"Execution Time - Parallel Training: {end_parallel - start_parallel:.2f} seconds")
print(f"Execution Time - Sequential Training: {end_sequential - start_sequential:.2f} seconds")
