---
title: "Selección del mejor modelo"
date: 2022-03-18T11:28:41+01:00
tags: [overfitting, underfitting, validación cruzada]
categories: [tutoriales]
---

En este post proporcionaremos una introducción intuitiva a los conceptos fundamentales de **overfitting** y **underfitting** en machine learning. Los modelos de machine learning nunca pueden hacer predicciones perfectas: el error de prueba nunca es exactamente cero. Esta carencia proviene del equilibrio fundamental entre la flexibilidad de modelado y el tamaño limitado del dataset de entrenamiento.

En un primer momento definiremos ambos problemas y caracterizaremos cómo y por qué surgen.

Posteriormente presentaremos una metodología para cuantificar estos problemas contrastando el error de entrenamiento con el error de prueba para varias opciones de la familia de modelos, los parámetros del modelo. Más importante aún, enfatizaremos el impacto del tamaño del dataset de entrenamiento en este equilibrio.

Por último, relacionaremos overfitting y underfitting a los conceptos de varianza y sesgo (bias) estadísticos.

# Framework de validación cruzada

En posts anteriores vimos algunos conceptos relacionados con la evaluación de modelos predictivos. Ahora vamos a analizar algunos detalles del framework de validación cruzada. Antes de ir a ello, vamos a detenernos en las razones de tener siempre conjuntos de entrenamiento y prueba. En primer lugar, echemos un vistazo a la limitación de usar un dataset sin excluir ninguna muestra.

Para ello vamos a usar el dataset de propiedades de California.


```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X, y = housing.data.copy(), housing.target.copy()
```

En este dataset, el objetivo es predecir el valor medio de las casas en un área de California. Las feautures recopiladas se basan en el mercado de la propiedad y en información geográfica. En este caso, el objetivo a predecir es una variable continua. Por tanto, es una tarea de regresión. Usaremos una modelo predictivo específico de regresión.


```python
print(housing.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 20640
    
        :Number of Attributes: 8 numeric, predictive attributes and the target
    
        :Attribute Information:
            - MedInc        median income in block
            - HouseAge      median house age in block
            - AveRooms      average number of rooms
            - AveBedrms     average number of bedrooms
            - Population    block population
            - AveOccup      average house occupancy
            - Latitude      house block latitude
            - Longitude     house block longitude
    
        :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    http://lib.stat.cmu.edu/datasets/
    
    The target variable is the median house value for California districts.
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    
    


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>



Para simplificar la visualización, vamos a transformar los precios del rango de cien mil dólares al rango de mil dólares.


```python
y *= 100
y.head()
```




    0    452.6
    1    358.5
    2    352.1
    3    341.3
    4    342.2
    Name: MedHouseVal, dtype: float64



## Error de entrenamiento vs error de prueba

Para resolver esta tarea de regresión usaremos un arbol de decisión de regresión.


```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)
```




    DecisionTreeRegressor(random_state=42)



Después de entrenar el regresor, nos gustaría saber su potencial rendimiento de generalización una vez lo despleguemos en producción. Para ello, usaremos el error medio absoluto que nos proporciona un error en las mismas unidades del objetivo, es decir, en miles de dólares.


```python
from sklearn.metrics import mean_absolute_error

y_predicted = regressor.predict(X)
score = mean_absolute_error(y, y_predicted)
print(f"De media, nuestro regresor comete un error de {score:.2f} k$")
```

    De media, nuestro regresor comete un error de 0.00 k$
    

Obtenemos una predicción perfecta sin errores. Esto es demasiado optimista y casi siempre pone de manifiesto un problema metodológico cuando hacemos machine learning. De hecho, entrenamos y predecimos en el mismo dataset. Dado que nuestro árbol de decisión creció por completo, cada instancia del dataset está almacenada en un nodo hoja. Por tanto, nuestro árbol de decisión ha memorizado completamente el dataset durante el `fit` y, en consecuencia, no comete ningún error cuando predice.

Este error calculado anteriormente se denomina **error empírico** o **error de entrenamiento**.

Entrenamos un modelo predictivo para minimizar el error de entrenamiento pero nuestro objetivo es minimizar el error en los datos que no se han visto durante el entrenamiento. Este error se llama también **error de generalización** o el "verdadero" **error de prueba**.

De esta forma, la evaluación más básica supone:

+ dividir nuestro dataset en dos subconjuntos: un conjunto de entrenamiento y un conjunto de prueba;
+ entrenar el modelo en el conjunto de entrenamiento;
+ estimar el error de entrenamiento en el conjunto de entrenamiento;
+ estimar el error de prueba en el conjunto de prueba.

Vamos a dividir nuestro dataset.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)
```

Ahora lo entrenamos.


```python
regressor.fit(X_train, y_train)
```




    DecisionTreeRegressor(random_state=42)



Finalmente, vamos a estimar los diferentes tipos de error. Empecemos calculando el error de entrenamiento.


```python
y_predicted = regressor.predict(X_train)
score = mean_absolute_error(y_train, y_predicted)
print(f"El error de entrenamiento de nuestro modelo es {score:.2f} k$")
```

    El error de entrenamiento de nuestro modelo es 0.00 k$
    

Observamos el mismo fenómeno que anteriormente: nuestro modelo memoriza el conjunto de entrenamiento. Sin embargo, vamos a calcular el error de prueba.


```python
y_predicted = regressor.predict(X_test)
score = mean_absolute_error(y_test, y_predicted)
print(f"El error de prueba de nuestro modelo es {score:.2f} k$")
```

    El error de prueba de nuestro modelo es 46.33 k$
    

Este es el error que realmente cabría esperar de nuestro modelo si lo pusiéramos en un entorno de producción.

## Estabilidad de las estimaciones de validación cruzada

Cuando hacemos una única división entrenamiento-prueba no damos ninguna indicación de la robustez de la evaluación de nuestro modelo predictivo: en particular, si el conjunto de prueba es pequeño, esta estimación del error de prueba será inestable y podría no reflejar la "verdadera tasa de error" que observaríamos con el mismo modelo en una cantidad ilimitada de datos de prueba.

Por ejemplo, podríamos haber tenido suerte cuando hicimos nuestra división aleatoria de nuestro limitado dataset y aislar algunos de los casos más fáciles de predecir del conjunto de prueba solo por casualidad: en este caso, la estimación del error de prueba sería demasiado optimista.

La **validación cruzada** permite estimar la solidez de un modelo predictivo repitiendo el procedimiento de división. Proporcionará varios errores de entrenamiento y prueba y, por tanto, alguna estimación de la variabilidad del rendimiento de generalización del modelo.

Existen diferentes [estrategias de validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators). Por el momento, nos centraremos en una llamada "*shuffle-split*". En cada iteración de esta estrategia:

+ mezclamos aleatoriamente el orden de las instancias de una copia del dataset;
+ dividimos el dataset mezclado en un conjunto de entrenamiento y uno de prueba;
+ entrenamos un nuevo modelo en el conjunto de entrenamiento;
+ evaluamos el error de prueba en el conjunto de prueba.

Repetimos este procedimiento `n_splits` veces. Tengamos en mente que el coste computacional se incrementa con `n_splits`.

![](/images/cross_validation_shufflesplit.png)

Este diagrama muestra el caso particular de la estrategia **shuffle-split** de validación cruzada usando `n_splits=5`. Por cada división de validación cruzada el procedimiento entrena un modelo en todos los ejemplo rojos y evalúa la puntuación del modelo en los ejemplos azules.

En este caso estableceremos `n_splits=40`, lo que significa que entrenaremos 40 modelos en total y todos ellos serán descartados: solo registraremos el rendimiento de generalización de cada variante en el conjunto de prueba.

Para evaluar el rendimiento de generalización de nuestro regresor podemos usar `sklearn.model_selection.cross_validate` con un objeto `sklearn.model_selection.ShuffleSplit`:


```python
from sklearn.model_selection import  cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=42)
cv_results = cross_validate(
    regressor, X, y, cv=cv, scoring="neg_mean_absolute_error"
)
```


```python
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.133615</td>
      <td>0.003504</td>
      <td>-47.329969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.130112</td>
      <td>0.003503</td>
      <td>-45.871795</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.134116</td>
      <td>0.003002</td>
      <td>-46.721323</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.132114</td>
      <td>0.003003</td>
      <td>-46.637444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.129111</td>
      <td>0.003002</td>
      <td>-46.978982</td>
    </tr>
  </tbody>
</table>
</div>



Una puntuación es una métrica donde cuanto más grande sea su valor mejores resultados. Por el contrario, un error es una métrica donde cuanto más pequeño sea su valor mejores resultados. El parámetro `scoring` in `cross_validate` siempre esepra una función que es una puntuación.

Para hacerlo fácil, todas las métricas de errores en scikit-learn, como `mean_absolute_error`, se pueden transformar en una puntuación para ser usadas en `cross_validate`. Para hacerlo necesitamos pasar el nombre de la métrica de error con el prefijo `neg_`. Por ejemplo, `scoring="neg_mean_absolute_error"`. En este caso, el negativo del error medio absoluto calculado equivaldría a una puntuación.

Vamos a revertir la negación para obtener el error real:


```python
cv_results["test_error"] = -cv_results["test_score"]
cv_results.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_score</th>
      <th>test_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.133615</td>
      <td>0.003504</td>
      <td>-47.329969</td>
      <td>47.329969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.130112</td>
      <td>0.003503</td>
      <td>-45.871795</td>
      <td>45.871795</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.134116</td>
      <td>0.003002</td>
      <td>-46.721323</td>
      <td>46.721323</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.132114</td>
      <td>0.003003</td>
      <td>-46.637444</td>
      <td>46.637444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.129111</td>
      <td>0.003002</td>
      <td>-46.978982</td>
      <td>46.978982</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.132614</td>
      <td>0.003504</td>
      <td>-45.130082</td>
      <td>45.130082</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.131113</td>
      <td>0.003503</td>
      <td>-47.191726</td>
      <td>47.191726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.130612</td>
      <td>0.003503</td>
      <td>-45.808697</td>
      <td>45.808697</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.131613</td>
      <td>0.003503</td>
      <td>-45.814624</td>
      <td>45.814624</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.133115</td>
      <td>0.003002</td>
      <td>-46.106001</td>
      <td>46.106001</td>
    </tr>
  </tbody>
</table>
</div>



Obtenemos información del tiempo de entrenamiento y predicción de cada iteración de validación cruzada. También obtenemos la puntuación de prueba que corresponde al error de prueba de cada división.


```python
len(cv_results)
```




    40



Obtenemos 40 entradas en nuestro dataframe resultante debido a las 40 divisiones realizadas. Por lo tanto, podemos mostrar la distribución del error de prueba y, así, tener una estimación de su variabilidad.


```python
import matplotlib.pyplot as plt

cv_results["test_error"].plot.hist(bins=10, edgecolor="black")
plt.xlabel("Error medio absoluto (k$)")
_ = plt.title("Distribución del error de prueba")
```


    
![png](/images/output_33_0.png)
    


Observamos que el error de prueba se agrupa en torno a 47 k y un rango de entre 45 k y 48.5 k.


```python
print(f"El error medio de validación cruzada es: "
      f"{cv_results['test_error'].mean():.2f} k$")
```

    El error medio de validación cruzada es: 46.53 k$
    


```python
print(f"La desviación típica de validación cruzada es: "
      f"{cv_results['test_error'].std():.2f} k$")
```

    La desviación típica de validación cruzada es: 0.83 k$
    

Observemos que la desviación típica es mucho más pequeña que la media. Podemos resumirlo como que nuestra estimación de validación cruzada del error de prueba es de 46.53 +/- 0.83 k$. Si tuviéramos que entrenar un único modelo en el dataset completo (sin validación cruzada) y luego después tuviéramos acceso a una cantidad ilimitada de datos de prueba, cabría esperar que el error de prueba verdadero cayera dentro de esa región.

Aunque esta información es interesante por sí misma, debería ser contrastada con la escala de la variabilidad natural del vector `objetivo` de nuestro dataset. Vamos a dibujar la distribución de esta variable objetivo:


```python
y.plot.hist(bins=20, edgecolor="black")
plt.xlabel("Valor medio de la vivienda (k$)")
_ = plt.title("Distribución del objetivo")
```


    
![png](/images/output_38_0.png)
    



```python
print(f"La desviación típica del objetivo es: {y.std():.2f} k$")
```

    La desviación típica del objetivo es: 115.40 k$
    

El rango de la variable objetivo varía desde cercano a 0 hasta 500, con una desviación típica de 115. Remarquemos que la media estimada del error de prueba obtenido por validación cruzada es un poco más pequeño que la escala natural de variación de la variable objetivo. Además, la desviación típica de la validación cruzada estimada del error de prueba es incluso más pequeña. Esto es un buen comienzo, pero no necesariamente suficiente para decidir si el rendimiento de generalización es suficientemente bueno para que nuestra predicción sea útil en la práctica.

Recordemos que nuestro modelo tiene, de media, un error de alrededor de 47 k. Con esta información y mirando la distribución del objetivo, tal error podría ser aceptable cuando predecimos viviendas con un valor de 500 k. Sin embargo, sería un problema con una vivienda con un valor de 50 k. Por tanto, esto indica que nuestra métrica (Error Absoluto Medio) no es ideal.

En su lugar podríamos elegir una métrica relativa al valor del objetivo a predecir: el error porcentual absoluto medio habría sido una mejor opción. Pero en todos los casos, un error de 47 k podría ser demasiado grande para usar automáticamente nuestro modelo para etiquetar viviendas sin la supervisión de un experto.

## Más detalles sobre `cross_validate`

Durante la validación cruzada, se entrenan y evalúan muchos modelos. De hecho, el número de elementos de cada matriz de salida de `cross_validate` es el resultado de uno de estos procedimientos `fit` / `score`. Para hacer explícito, es posible recuperar estos modelos entrenados para cada una de las divisiones/particiones pasando la opción `return_estimator=True` en `cross_validate`.


```python
cv_results = cross_validate(regressor, X, y, return_estimator=True)
cv_results
```




    {'fit_time': array([0.15413189, 0.15012884, 0.15012932, 0.15063   , 0.15162921]),
     'score_time': array([0.002002  , 0.00250196, 0.00250196, 0.00200152, 0.00250244]),
     'estimator': (DecisionTreeRegressor(random_state=42),
      DecisionTreeRegressor(random_state=42),
      DecisionTreeRegressor(random_state=42),
      DecisionTreeRegressor(random_state=42),
      DecisionTreeRegressor(random_state=42)),
     'test_score': array([0.28326244, 0.4226389 , 0.45552292, 0.23727262, 0.41430376])}




```python
cv_results["estimator"]
```




    (DecisionTreeRegressor(random_state=42),
     DecisionTreeRegressor(random_state=42),
     DecisionTreeRegressor(random_state=42),
     DecisionTreeRegressor(random_state=42),
     DecisionTreeRegressor(random_state=42))



Los cinco regresores de árbol de decisión corresponden a los cinco árboles de decisión entrenados en las diferentes particiones. Tener acceso a estos regresores es útil porque permite inspeccionar los parametros entrenados internos de estos regresores.

En el caso de solo estemos interesados en la puntuación de prueba, scikit-learn provee una función `cross_val_score`. Es idéntica a llamar a la función `cross_validate` y seleccionar solo `test_score`.


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, X, y)
scores
```




    array([0.28326244, 0.4226389 , 0.45552292, 0.23727262, 0.41430376])




```python

```
