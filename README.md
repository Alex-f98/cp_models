# CP Models: Modelos de Predicción Conformal

Un paquete de Python para modelos de predicción conformal que proporciona cuantificación de incertidumbre para predicciones de machine learning.

## Resumen General

La predicción conformal es un marco estadístico que proporciona intervalos/conjuntos de predicción válidos con cobertura garantizada bajo supuestos mínimos. Este paquete implementa métodos conformales de división (split conformal) para tareas tanto de clasificación como de regresión.

## Instalación

```bash
pip install -e .
```

## Dependencias

- numpy>=1.24.0
- scikit-learn>=1.3.0
- torch>=2.0.0
- torchvision>=0.15.0
- matplotlib>=3.5.0

## Estructura del Paquete

```
cp_models/
├── classification/          # Métodos de clasificación conformal
│   ├── __init__.py
│   └── split_conformal.py   # Clasificador conformal de división
├── regression/             # Métodos de regresión conformal
│   ├── __init__.py
│   └── split_conformal_regressor.py  # Regresor conformal de división
├── models/                 # Modelos de redes neuronales
│   ├── cnn/               # Redes neuronales convolucionales
│   ├── mlp/               # Perceptrones multicapa
│   └── utils/             # Utilidades de modelos
├── scores/                # Funciones de puntuación de conformidad
│   ├── absolute_score.py  # Puntuaciones de residuos absolutos
│   └── squared_score.py   # Puntuaciones de residuos cuadrados
├── metrics/               # Métricas de evaluación
├── data/                  # Utilidades de carga de datos
└── utils/                 # Utilidades generales
```

## Componentes Principales

### Clasificación

#### SplitConformalClassifier

Un predictor conformal de división para tareas de clasificación que proporciona conjuntos de predicción con cobertura garantizada.

```python
from cp_models import SplitConformalClassifier
from cp_models.models.mlp import GenericMLP

# Crear modelo base
model = GenericMLP(input_dim=784, num_classes=10, epochs=5)

# Crear clasificador conformal
cp = SplitConformalClassifier(model, alpha=0.05)

# Ajustar y calibrar
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)

# Hacer predicciones con incertidumbre
prediction_sets = cp.predict_set(X_test)
single_predictions = cp.predict(X_test)
probabilities = cp.predict_proba(X_test)
```

**Métodos Clave:**
- `fit(X_train, y_train)`: Entrenar el modelo base
- `calibrate(X_cal, y_cal)`: Calcular puntuaciones de conformidad en el conjunto de calibración
- `predict(X_test)`: Hacer predicciones puntuales
- `predict_set(X_test)`: Hacer conjuntos de predicción
- `predict_proba(X_test)`: Obtener probabilidades de clases
- `predict_interval(X_test)`: Obtener intervalos de predicción

### Regresión

#### SplitConformalRegressor

Un predictor conformal de división para tareas de regresión que proporciona intervalos de predicción con cobertura garantizada.

```python
from cp_models import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor

# Crear modelo base
model = RandomForestRegressor(n_estimators=100)

# Crear regresor conformal
cp = SplitConformalRegressor(model, alpha=0.1, score=AbsoluteScore())

# Ajustar y calibrar
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)

# Hacer predicciones con incertidumbre
prediction_intervals = cp.predict_interval(X_test)
point_predictions = cp.predict(X_test)
```

**Métodos Clave:**
- `fit(X_train, y_train)`: Entrenar el modelo base
- `calibrate(X_cal, y_cal)`: Calcular puntuaciones de conformidad en el conjunto de calibración
- `predict(X_test)`: Hacer predicciones puntuales
- `predict_interval(X_test)`: Obtener intervalos de predicción

### Puntuaciones de Conformidad

El paquete proporciona diferentes funciones de puntuación de conformidad para regresión:

#### AbsoluteScore
Calcula residuos absolutos: |y - y_pred|

```python
from cp_models import AbsoluteScore

score = AbsoluteScore()
residuals = score(y_true, y_pred)
```

#### SquaredScore
Calcula residuos cuadrados: (y - y_pred)²

```python
from cp_models import SquaredScore

score = SquaredScore()
residuals = score(y_true, y_pred)
```

### Modelos de Redes Neuronales

#### GenericMLP
Perceptrón multicapa para tareas de clasificación.

```python
from cp_models.models.mlp import GenericMLP

model = GenericMLP(
    input_dim=784,
    num_classes=10,
    hidden_dims=[256, 128],
    epochs=10,
    batch_size=64,
    learning_rate=0.001
)
```

#### GenericCNN
Red neuronal convolucional para clasificación de imágenes.

```python
from cp_models.models.cnn import GenericCNN

model = GenericCNN(
    input_channels=1,
    num_classes=10,
    conv_channels=[32, 64],
    kernel_sizes=[3, 3],
    epochs=10,
    batch_size=64
)
```

### Utilidades de Datos

El paquete proporciona utilidades para cargar conjuntos de datos comunes:

```python
from cp_models.models.utils import get_data

# Cargar datos MNIST
X_train, y_train, X_test, y_test, X_cal, y_cal = get_data(
    source="mnist",
    flatten=False,
    size_calib=50
)

# Cargar datos Fashion-MNIST
X_train, y_train, X_test, y_test, X_cal, y_cal = get_data(
    source="fashion_mnist",
    flatten=True,
    size_calib=100
)
```

## Ejemplos de Uso

### Ejemplo de Clasificación

```python
import torch
from cp_models import SplitConformalClassifier
from cp_models.models.mlp import GenericMLP
from cp_models.models.utils import get_data

# Cargar datos
X_train, y_train, X_test, y_test, X_cal, y_cal = get_data(
    source="mnist", 
    flatten=False, 
    size_calib=50
)

# Aplanar para MLP
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_cal = X_cal.reshape(X_cal.shape[0], -1)

# Crear y entrenar clasificador conformal
model = GenericMLP(input_dim=784, num_classes=10, epochs=5)
cp = SplitConformalClassifier(model, alpha=0.05)

cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)

# Hacer predicciones
pred_sets = cp.predict_set(X_test)
y_pred = cp.predict(X_test)
probabilities = cp.predict_proba(X_test)

# Evaluar cobertura
coverage = []
for i, pred_set in enumerate(pred_sets):
    coverage.append(y_test[i] in pred_set)
print(f"Cobertura: {sum(coverage)/len(coverage):.3f}")
```

### Ejemplo de Regresión

```python
from cp_models import SplitConformalRegressor, AbsoluteScore
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generar datos sintéticos
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Dividir datos
n_train, n_cal = 600, 200
X_train, y_train = X[:n_train], y[:n_train]
X_cal, y_cal = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
X_test, y_test = X[n_train+n_cal:], y[n_train+n_cal:]

# Crear y entrenar regresor conformal
model = RandomForestRegressor(n_estimators=100, random_state=42)
cp = SplitConformalRegressor(model, alpha=0.1, score=AbsoluteScore())

cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)

# Hacer predicciones
intervals = cp.predict_interval(X_test)
y_pred = cp.predict(X_test)

# Evaluar cobertura
coverage = ((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])).mean()
print(f"Cobertura: {coverage:.3f}")
```

## Características Principales

- **Cobertura Válida**: Proporciona intervalos/conjuntos de predicción con cobertura marginal garantizada
- **Agnóstico al Modelo**: Funciona con cualquier modelo subyacente (sklearn, PyTorch, etc.)
- **Flexible**: Soporta diferentes puntuaciones de conformidad y métodos de calibración
- **Fácil Integración**: Compatible con la API de scikit-learn
- **Soporte de Redes Neuronales**: Modelos incorporados para tareas comunes de deep learning

## Teoría

La predicción conformal divide los datos en:
1. **Conjunto de entrenamiento**: Usado para ajustar el modelo subyacente
2. **Conjunto de calibración**: Usado para calcular puntuaciones de conformidad y cuantiles
3. **Conjunto de prueba**: Usado para evaluación

Para un nivel de significancia α, el método asegura que:
- Clasificación: P(Y ∈ conjunto de predicción) ≥ 1 - α
- Regresión: P(Y ∈ intervalo de predicción) ≥ 1 - α

## Licencia

Este paquete es desarrollado por Brian Fuentes (bfuentes@fi.uba.ar) para investigación y aplicaciones de predicción conformal.