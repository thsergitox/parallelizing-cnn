## Paralelizando el entrenamiento de Redes Convolucionales 

Este proyecto busca **comparar** el enfoque **paralelo** y el **secuencial** para entrenar un **clasificador tipo MLP (Multi-Layer Perceptron)** usando el dataset MNIST. A nivel conceptual, se pretende ilustrar cómo **distribuir el entrenamiento** en varios subconjuntos de datos y observar los **beneficios en tiempo de cómputo**, así como los **trade-offs** en rendimiento y precisión.

---

## **Índice**
- [Paralelizando el entrenamiento de Redes Convolucionales](#paralelizando-el-entrenamiento-de-redes-convolucionales)
- [**Índice**](#índice)
- [**Descripción General**](#descripción-general)
- [**Estructura de Archivos**](#estructura-de-archivos)
- [**Instalación y Dependencias**](#instalación-y-dependencias)
- [**Estrategia de Paralelización \& Relación con el Paper**](#estrategia-de-paralelización--relación-con-el-paper)
  - [**Lo que hace el Paper**](#lo-que-hace-el-paper)
  - [**Lo que hacemos en este Proyecto**](#lo-que-hacemos-en-este-proyecto)
- [**Descripción Detallada del Código**](#descripción-detallada-del-código)
- [**Speed-Up y Fórmulas**](#speed-up-y-fórmulas)
  - [**Teoría Simplificada del Speed-Up (Amdahl / Gustafson)**](#teoría-simplificada-del-speed-up-amdahl--gustafson)
- [**Resultados y Discusión (Trade-Offs)**](#resultados-y-discusión-trade-offs)
  - [**¿Por qué no coincide la precisión?**](#por-qué-no-coincide-la-precisión)
  - [**Trade-Offs**](#trade-offs)
- [**Conclusiones**](#conclusiones)

---

## **Descripción General**

El proyecto **divide el conjunto de entrenamiento** (MNIST) en 4 subconjuntos y entrena **4 redes MLP diferentes** de manera **paralela**, utilizando la librería `joblib`. La combinación de los 4 modelos no es una concatenación de pesos, sino que se realiza un **ensemble por votación mayoritaria** en la etapa de inferencia (predicción).

Por otro lado, se entrena **un modelo secuencial** que usa todo el dataset de una sola vez como **baseline**. Se registran los tiempos de entrenamiento para comparar la **aceleración** obtenida en modo paralelo (speed-up) contra el modo secuencial.

Finalmente, se presenta la comparación de **matrices de confusión** y **métricas de clasificación** (precision, recall, f1-score) para evaluar la **calidad del modelo**.

---

## **Estructura de Archivos**

- `README.md`: Este archivo con la explicación detallada del proyecto.
- `entrenamiento_mlp.py` (o nombre equivalente): El script principal que contiene el código para entrenamiento paralelo y secuencial, la evaluación y las gráficas.

---

## **Instalación y Dependencias**

1. **Python 3.x**  
2. Librerías:
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `networkx`
   - `joblib`
   - `tensorflow` / `keras` (para cargar MNIST con `mnist.load_data()`)
3. (Opcional) **GPU NVIDIA** con CUDA, para mayores beneficios en redes más complejas (CNN con TensorFlow/PyTorch), aunque en este ejemplo nos basamos en CPU.

**Instalación**:

Para este proeycto se debe crear un python enviroment e inicializarlo

```
python3 -m venv venv

# Si se encuentra en linux
source venv/bin/activate

# Si se encuentra en Windows
.\venv\Scripts\activate
```

Una vez dentro de nuestro enviroment instalar las dependencias.


```bash
pip install -r requirements.txt
```



A continuación, ejecutar el main.py 

``` sh
python main.py
```



ó inicializar un servidor de jupyter notebook, para su ejecución correr el siguiente comando

```sh
jupyter-notebook
```

Y entrar al archivo main-notebook.ipynb


---

## **Estrategia de Paralelización & Relación con el Paper**

### **Lo que hace el Paper**
- **Contexto**: El paper describe un enfoque para **paralelizar el entrenamiento de una CNN** en un sistema distribuido, solapando comunicación y cómputo.  
- **Estrategia clave**: Cada nodo de cómputo (worker) entrena gradientes parciales, y mientras va computando los modelos, un hilo independiente **envía** los gradientes ya calculados a los otros nodos o al nodo maestro, **maximizando el solapamiento** de comunicación y cómputo.

### **Lo que hacemos en este Proyecto**
- Se entrenan 4 **modelos MLP independientes** (cada uno con su propio subconjunto de datos) en **paralelo** usando `joblib`.
- **No hay fusión de pesos** entre los 4 modelos.
- Cada modelo produce predicciones, que se **combinan vía votación mayoritaria** (ensemble) para la inferencia.
- Esto **reduce el tiempo total de entrenamiento** ya que en lugar de entrenar un solo modelo con todos los datos de manera secuencial, se **distribuye la carga** en 4 *jobs* simultáneos.  


---

## **Descripción Detallada del Código**


```python
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
```

**Nota Importante**:  
- **No** se concatenan los pesos de los modelos entrenados en paralelo. Cada uno tiene sus propios parámetros. El ensamblado final ocurre **solo** a nivel de **predicciones**, no a nivel de la red neuronal en sí.

---

## **Speed-Up y Fórmulas**

Para cuantificar la **aceleración** lograda por el enfoque paralelo, solemos usar el concepto de **Speed-Up** clásico:

\[
\text{Speed-Up} = \frac{T_{\text{secuencial}}}{T_{\text{paralelo}}}
\]

donde:
- \( T_{\text{secuencial}} \) es el tiempo que demora entrenar el modelo de manera secuencial.
- \( T_{\text{paralelo}} \) es el tiempo del entrenamiento paralelo.

En nuestro caso, de acuerdo con los resultados, por ejemplo:
- \( T_{\text{secuencial}} \approx 9.19 \) segundos.
- \( T_{\text{paralelo}} \approx 6.42 \) segundos.

Entonces,

\[
\text{Speed-Up} \approx \frac{9.19}{6.42} \approx 1.43 \; .
\]

Por lo que se fue un $43$% más rápido. 

### **Teoría Simplificada del Speed-Up (Amdahl / Gustafson)**
- **Amdahl's Law** sugiere que hay partes del algoritmo que no se pueden paralelizar, por lo que el speed-up nunca es lineal (nunca llegaremos a 4× speed-up exacto con 4 procesos, debido a la porción secuencial irreducible y la sobrecarga de comunicación).
- **Gustafson's Law** argumenta que, a medida que crece el tamaño del problema, la parte paralelizable aumenta y la secuencial se mantiene casi constante. Esto podría permitir una escalabilidad mejor con más datos o más procesos.

En nuestro ejemplo, la estrategia paralela también conlleva un **overhead** en iniciar los hilos (o procesos) y luego combinar predicciones.

---

## **Resultados y Discusión (Trade-Offs)**

A partir de la **matriz de confusión** y el **reporte de clasificación** se observa que:

- **Enfoque Paralelo**:  
  - **Precisión Global**: ~0.95  
  - **Tiempo de Entrenamiento**: ~6.42 segundos  

- **Enfoque Secuencial**:  
  - **Precisión Global**: ~0.98  
  - **Tiempo de Entrenamiento**: ~9.19 segundos  

### **¿Por qué no coincide la precisión?**
- El **modelo secuencial** entrena con **todas las muestras** directamente, ajustando sus pesos de forma coherente en cada iteración.  
- En cambio, el **enfoque paralelo** entrena **4 modelos diferentes** con subconjuntos distintos. Cada modelo ve solo una fracción de los datos. Aunque combinamos sus predicciones en la etapa de inferencia (ensemble), la precisión del ensemble puede ser algo inferior a la de un modelo entrenado globalmente.

### **Trade-Offs**  
1. **Velocidad vs. Precisión**: El enfoque paralelo es **más rápido** pero puede sacrificar algo de precisión, dado que cada modelo no “ve” todo el dataset en un solo proceso de entrenamiento.  
2. **Recursos Computacionales**: Para que el enfoque paralelo brinde beneficios, necesitamos **múltiples núcleos** (CPU) o un cluster. Con una sola CPU puede incluso empeorar el tiempo debido al overhead.  
3. **Escalabilidad**: A medida que aumente el número de subconjuntos (o nodos de cómputo en un entorno distribuido), se incrementa la complejidad de coordinación (comunicación de parámetros, configuración de software).  
4. **Ensamblado Final (Ensemble)**: No fusionamos parámetros. Combinar pesos entre redes individuales entrenadas separadamente **no es trivial** y puede ocasionar inconsistencias. Optamos por el **voto mayoritario**, un método simple y efectivo en muchos casos.

---

## **Conclusiones**

1. **Enfoque Paralelo**:  
   - Ofrece **reducción del tiempo** de entrenamiento (speed-up ~1.43 en este caso).
   - Suele escalar si la parte paralelizable es grande en comparación con la parte secuencial.  

2. **Enfoque Secuencial**:  
   - Tiende a **mayor exactitud** al ver todos los datos de forma unificada.
   - Se hace más lento a medida que el tamaño del dataset crece y no se aprovechan múltiples núcleos o GPU.

3. **Aplicabilidad**:  
   - Para proyectos con volúmenes de datos muy grandes, **paralelizar** o **distribuir** el entrenamiento es esencial para reducir tiempos.
   - Este ejemplo con MLPClassifier y `joblib` es un acercamiento didáctico; en un escenario real de **CNN** con millones de parámetros y grandes volúmenes de datos, se podría usar la **estrategia de solapamiento comunicación-cómputo** del paper, escalando a varios nodos con GPU.  

4. **Trade-Off**:  
   - Un **ensemble** de 4 MLPs entrenados en subconjuntos mejora el tiempo total, pero la precisión final puede ser menor que la del modelo único entrenado secuencialmente con todo el dataset.  

**Comentarios finales**:  
- Si **contamos con una GPU NVIDIA** y frameworks como PyTorch o TensorFlow, el entrenamiento de CNNs puede volverse drásticamente más rápido.  
- Para implementar la estrategia descrita en el *paper* (overlap de comunicación y cómputo de gradientes), se requeriría un entorno distribuido (MPI, Horovod, etc.) donde cada nodo comparta gradientes de manera eficiente.  

<br>

---
**Autor**: *Sergio Pezo*  
**Contacto**: *sergio.pezo.j@uni.pe*
**Fecha**: *13/12/2024*  

---