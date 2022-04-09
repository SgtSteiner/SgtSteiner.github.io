---
title: "Ajuste de hiperparámetros"
date: 2022-03-23T12:31:47+01:00
tags: [hiperparámetros, grid-search, randomized-search, nested-cross-validation, parallel coordinates, ]
categories: [tutoriales]
---

En posts anteriores vimos cómo crear, entrenar, predecir e incluso evaluar un modelo predictivo. Sin embargo, no cambiamos ninguno de los parámetros del modelo que tenemos a nuestra disposición cuando creamos una instancia. Por ejemplo, para k-nearest neighbors, inicialmente usamos los parámetros por defecto: `n_neighbors=5` antes de probar otros parámetros del modelo.

Estos parámetros se denominan **hiperparámetros**: son parámetros usados para controlar el proceso de aprendizaje, por ejemplo el parámetro `k` de k-nearest neighbors. Los hiperparámetros son especificados por el usuario, a menudo ajustados manualmente (o por una búsqueda automática exhaustiva) y no pueden ser estimados a partir de los datos. No deben confundirse con los otros parámetros que son inferidos durante el proceso de entrenamiento. Estos parámetros definen el modelo en sí mismo, por ejemplo `coef_` para los modelos lineales.

En este post mostraremos en primer lugar que los hiperparámetros tienen un impacto en el rendimiento del modelo y que los valores por defecto no son necesariamente la mejor opción. Posteriormente, mostraremos cómo definir hiperparámetros en un modelo de scikit-learn. Por último, mostraremos estrategias que nos permitirán seleccionar una combinación de hiperparámetros que maximicen el rendimiento del modelo.

En concreto repasaremos los siguientes aspectos:

+ cómo usar `get_params` y `set_params` para obtener los parámetros de un modelo y establecerlos, respectivamente;
+ cómo optimizar los hiperparámetros de un modelo predictivo a través de grid-search;
+ cómo la búsqueda de más de dos hiperparámetros es demasiado costosa;
+ cómo grid-search no encuentra necesariamente una solución óptima;
+ cómo la búsqueda aleatoria ofrece una buena alternativa a grid-search cuando el número de parámetros a ajustar es más de dos. También evita la regularidad impuesta por grid-search que puede resultar problemática en ocasiones;
+ cómo evaluar el rendimiento predictivo de un modelo con hiperparámetros ajustados usando el procedimiento de validación cruzada anidada.

# Establecer y obtener hiperparámetros en scikit-learn

El proceso de aprendizaje de un modelo predictivo es conducido por un conjunto de parámetros internos y un conjunto de datos de entrenamiento. Estos parámetros internos se denominan hiperparámetros y son específicos de cada familia de modelos. Además, un conjunto específico de hiperparámetros es óptimo para un dataset específico y, por lo tanto, necesitan optimizarse.

Vamos a mostrar como podemos obtener y establecer el valor de un hiperparámetro en un estimador de scikit-learn. Recordemos que los hiperparámetros se refieren a los parámetros que controlarán el proceso de aprendizaje. No debemos confundirlos con los parámetros entrenados, resultado del entrenamiento. Estos parámetros entrenados se reconocen en scikit-learn porque tienen el sufijo `_`, por ejemplo, `model_coef_`.

Utilizaremos el dataset del [Censo US de 1944](http://www.openml.org/d/1590), del que únicamente usaremos las variables numéricas.


{{< highlight "python" "linenos=false">}}
import pandas as pd

adult_census = pd.read_csv("adult_census.csv")

target_name = "class"
y = adult_census[target_name]
data = adult_census.drop(columns=[target_name])

numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]
X = data[numerical_columns]
X.head()
{{< /highlight >}}




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
      <th>age</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



Vamos a crear un modelo predictivo simple compuesto por un scaler seguido por un clasificador de regresión logística.

Muchos modelos, incluidos los lineales, trabajan mejor si todas las características tienen un escalado similar. Para este propósito, usaremos un `StandardScaler`, que transforma los datos escalando las features.


{{< highlight "python" "linenos=false">}}
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

set_config(display="diagram")

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression())
])
{{< /highlight >}}

Podemos evaluar el rendimiento de generalización del modelo a través de validación cruzada.


{{< highlight "python" "linenos=false">}}
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, X, y)
scores = cv_results["test_score"]
print(f"Puntuación de precisión a través de validación cruzada:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
{{< /highlight >}}

    Puntuación de precisión a través de validación cruzada:
    0.800 +/- 0.003
    

Hemos creado un modelo con el valor por defecto de `C` que es igual a 1. Si quisiéramos usar un parámetro `C` distinto, podríamos haberlo hecho cuando creamos el objeto `LogisticRegression` con algo como `LogisticRegression(C=1e-3)`. También podemos cambiar el parámetro de un modelo después de que haya sido creado con el método `set_params`, disponible para todos los estimadores de scikit-learn. Por ejemplo, podemos establecer `C=1e-3`, entrenar y evaluar el modelo:


{{< highlight "python" "linenos=false">}}
model.set_params(classifier__C=1e-3)
cv_results = cross_validate(model, X, y)
print(f"Puntuación de precisión a través de validación cruzada:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
{{< /highlight >}}

    Puntuación de precisión a través de validación cruzada:
    0.800 +/- 0.003
    

Cuando el modelo está en un `Pipeline`, los nombres de los parámetros tiene la forma `<nombre_modelo>__<nombre_parámetro>`. En nuestro caso, `classifier` proviene de la definición del `Pipeline` y `C` es el nombre del parámetro de `LogisticRegression`.

Generalmente, podemos usar el método `get_params` en los modelos de scikit-learn para listar todos los parámetros con sus respectivos valores.


{{< highlight "python" "linenos=false">}}
model.get_params()
{{< /highlight >}}




    {'memory': None,
     'steps': [('preprocessor', StandardScaler()),
      ('classifier', LogisticRegression(C=0.001))],
     'verbose': False,
     'preprocessor': StandardScaler(),
     'classifier': LogisticRegression(C=0.001),
     'preprocessor__copy': True,
     'preprocessor__with_mean': True,
     'preprocessor__with_std': True,
     'classifier__C': 0.001,
     'classifier__class_weight': None,
     'classifier__dual': False,
     'classifier__fit_intercept': True,
     'classifier__intercept_scaling': 1,
     'classifier__l1_ratio': None,
     'classifier__max_iter': 100,
     'classifier__multi_class': 'auto',
     'classifier__n_jobs': None,
     'classifier__penalty': 'l2',
     'classifier__random_state': None,
     'classifier__solver': 'lbfgs',
     'classifier__tol': 0.0001,
     'classifier__verbose': 0,
     'classifier__warm_start': False}



`get_params` devuelve un diccionario cuyas claves son los nombres de los parámetros y sus valores los valores de dichos parámetros. Si queremos obtener el valor de un único parámetro, por ejemplo, `classifier__C` usamos lo siguiente:


{{< highlight "python" "linenos=false">}}
model.get_params()["classifier__C"]
{{< /highlight >}}




    0.001



Podemos variar sistemáticamente el valor de C para ver si existe un valor óptimo.


{{< highlight "python" "linenos=false">}}
for C in [1e-3, 1e-2, 1e-1, 1, 10]:
    model.set_params(classifier__C=C)
    cv_results = cross_validate(model, X, y)
    scores = cv_results["test_score"]
    print(f"Puntuación de precisión de validación cruzada con C={C}:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
{{< /highlight >}}

    Puntuación de precisión de validación cruzada con C=0.001:
    0.787 +/- 0.002
    Puntuación de precisión de validación cruzada con C=0.01:
    0.799 +/- 0.003
    Puntuación de precisión de validación cruzada con C=0.1:
    0.800 +/- 0.003
    Puntuación de precisión de validación cruzada con C=1:
    0.800 +/- 0.003
    Puntuación de precisión de validación cruzada con C=10:
    0.800 +/- 0.003
    

Podemos ver que mientras C sea lo suficientemente alto, el modelo parece rendir bien.

Lo que hemos hecho aquí es muy manual: implica recorrer los valores de C y seleccionar manualmente el mejor. Veremos cómo realizar esta tarea de forma automática.

Cuando evaluamos una familia de modelos en datos de prueba y seleccionamos el que mejor se ejecuta, no podemos confiar en la correpondiente precisión de la estimación y necesitamos aplicar el modelo en nuevos datos. De hecho, los datos de prueba se han usado para seleccionar el modelo y, por lo tanto, ya no es independiente de este modelo.

# Ajuste de hiperparámetros por *grid-search*

Vamos a mostrar cómo optimizar hiperparámetros usando el enfoque de grid-search.

Seguimos con nuestro dataset del censo.


{{< highlight "python" "linenos=false">}}
target_name = "class"
y = adult_census[target_name]
y.head()
{{< /highlight >}}




    0     <=50K
    1     <=50K
    2      >50K
    3      >50K
    4     <=50K
    Name: class, dtype: object



Vamos a eliminar de nuestro datos el objetivo y la columna `"education-num"`, dado que es información duplicada de la columna `"education"`.


{{< highlight "python" "linenos=false">}}
X = adult_census.drop(columns=[target_name, "education-num"])
X.head()
{{< /highlight >}}




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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>Private</td>
      <td>226802</td>
      <td>11th</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>89814</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Local-gov</td>
      <td>336951</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>Private</td>
      <td>160323</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>?</td>
      <td>103497</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
    </tr>
  </tbody>
</table>
</div>



Dividimos el dataset en entrenamiento y prueba.


{{< highlight "python" "linenos=false">}}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
{{< /highlight >}}

Vamos a definir un pipeline y manejaremos tanto las variables numéricas como las categóricas.


{{< highlight "python" "linenos=false">}}
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X)
{{< /highlight >}}

En este caso, estamos usando un modelo basado en árbol como un clasificador (es decir, `HistGradientBoostingClassifier`). Esto significa que:

+ las variables numéricas no necesitan escalado;
+ las variables categóricas se pueden manejar con un `OrdinalEncoder` incluso si el orden codificado no tiene sentido;
+ En los modelos basados en árbol, `OrdinalEncoder` evita tener representaciones de alta dimensionalidad.

Vamos a construir nuestro `OrdinalEncoder` pasándole las categorías conocidas.


{{< highlight "python" "linenos=false">}}
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
{{< /highlight >}}

Usaremos un `ColumnTransformer` para seleccionar las columnas categóricas y aplicarles el `OrdinalEncoder`.


{{< highlight "python" "linenos=false">}}
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("cat_preprocessor", categorical_preprocessor, categorical_columns)],
    remainder="passthrough", sparse_threshold=0)
{{< /highlight >}}

Por último, usaremos un clasificador de árbol (por ejemplo, *histogram gradient-boosting*) para predecir si una persona gana más de 50 k$ al año.


{{< highlight "python" "linenos=false">}}
from sklearn.ensemble import HistGradientBoostingClassifier

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42, max_leaf_nodes=4
    ))
])
model
{{< /highlight >}}




<style>#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 {color: black;background-color: white;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 pre{padding: 0;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-toggleable {background-color: white;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-estimator:hover {background-color: #d4ebff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-item {z-index: 1;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-parallel-item:only-child::after {width: 0;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-bfa4b090-bb0e-4721-afa1-b8867206ea58 div.sk-text-repr-fallback {display: none;}</style><div id="sk-bfa4b090-bb0e-4721-afa1-b8867206ea58" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1), [&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="820085ec-5b92-4bee-873e-6726debea34c" type="checkbox" ><label for="820085ec-5b92-4bee-873e-6726debea34c" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d4538111-be48-4cc7-af38-39fccedf57e2" type="checkbox" ><label for="d4538111-be48-4cc7-af38-39fccedf57e2" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="45ccddd5-1f11-42fd-b0e2-c40787a4b44c" type="checkbox" ><label for="45ccddd5-1f11-42fd-b0e2-c40787a4b44c" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f20179c5-b176-4e8b-a541-4377917be797" type="checkbox" ><label for="f20179c5-b176-4e8b-a541-4377917be797" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="66dee3d8-b3ff-4dbd-b15f-afd508e44956" type="checkbox" ><label for="66dee3d8-b3ff-4dbd-b15f-afd508e44956" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0e61c6de-dc28-4d07-8684-14e776da4bf0" type="checkbox" ><label for="0e61c6de-dc28-4d07-8684-14e776da4bf0" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c6b34bd5-2a99-4e27-8881-bb37f7939bce" type="checkbox" ><label for="c6b34bd5-2a99-4e27-8881-bb37f7939bce" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div>



Pasemos ahora al ajuste con grid-search. Anteriormente usamos un bucle `for` para cada hiperparámetro con el fin de encontrar la mejor combinación a partir de un conjunto de valores. La clase `GridSearchCV` de scikit-learn implementa una lógica muy similar con mucho menos código repetitivo. Vamos a ver cómo usar el estimador `GridSearchCV` para realizar esta búsqueda. Dado que grid-search puede ser costoso, únicamente exploraremos la combinación tasa de aprendizaje y máximo número de nodos.


{{< highlight "python" "linenos=false">}}
%%time
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),
    "classifier__max_leaf_nodes": (3, 10, 30)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=-1, cv=2)
model_grid_search.fit(X_train, y_train)
{{< /highlight >}}

    CPU times: total: 14.3 s
    Wall time: 5.51 s
    




<style>#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 {color: black;background-color: white;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 pre{padding: 0;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-toggleable {background-color: white;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-item {z-index: 1;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-parallel-item:only-child::after {width: 0;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-9b70fb2d-44a4-4963-afbb-8a2c6bac99a3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=2,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))]),n_jobs=-1,param_grid={&#x27;classifier__learning_rate&#x27;: (0.01, 0.1, 1, 10),&#x27;classifier__max_leaf_nodes&#x27;: (3, 10, 30)})</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="80e416b7-11ab-4602-b188-1dcd00dc20eb" type="checkbox" ><label for="80e416b7-11ab-4602-b188-1dcd00dc20eb" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=2,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))]),n_jobs=-1,param_grid={&#x27;classifier__learning_rate&#x27;: (0.01, 0.1, 1, 10),&#x27;classifier__max_leaf_nodes&#x27;: (3, 10, 30)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ca9506ba-28e6-4a36-8103-106d6e6d5e33" type="checkbox" ><label for="ca9506ba-28e6-4a36-8103-106d6e6d5e33" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0026db78-2776-471a-9600-d6f56a04f50d" type="checkbox" ><label for="0026db78-2776-471a-9600-d6f56a04f50d" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="97530e35-4ede-416e-affa-a4f7c3de14a2" type="checkbox" ><label for="97530e35-4ede-416e-affa-a4f7c3de14a2" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b526a3b7-55ec-448b-898e-f4cba1e3c48b" type="checkbox" ><label for="b526a3b7-55ec-448b-898e-f4cba1e3c48b" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a8fe2b9f-1537-467e-b6b4-096e4c01202d" type="checkbox" ><label for="a8fe2b9f-1537-467e-b6b4-096e4c01202d" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="15e715f3-9c90-493f-b50e-56dc2074a9f4" type="checkbox" ><label for="15e715f3-9c90-493f-b50e-56dc2074a9f4" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



Finalmente, comprobamos la precisión de nuestro modelo usando el conjunto de prueba.


{{< highlight "python" "linenos=false">}}
accuracy = model_grid_search.score(X_test, y_test)
print(f"La puntuación de precisión de prueba del pipeline grid-search es:"
      f"{accuracy:.2f}")
{{< /highlight >}}

    La puntuación de precisión de prueba del pipeline grid-search es:0.88
    

El estimador `GridSearchCV` toma una parámetro `param_grid` que define todos los hiperparámetros y sus valores asociados. Grid-search se encargará de crear todas las posibles combinaciones y probarlas.

El número de combinaciones será igual al producto del número de valores a explorar para cada parámetros (es decir, en nuestro ejemplo 4 x 3 combinaciones). Por tanto, añadir nuevos parámetros con sus valores asociados a ser explorados se vuelve rápidamente computacionalmente costoso.

Una vez que grid-search es entrenado, se puede usar como cualquier otro predictor llamando a sus métodos `predict` y `predict_proba`. Internamente, usará el modelo con los mejores parámetros encontrados durante el `fit`.

Vamos a obtener las predicciones para los primeros 5 ejemplos usando el estimador con los mejores parámetros.


{{< highlight "python" "linenos=false">}}
model_grid_search.predict(X_test.iloc[0:5])
{{< /highlight >}}




    array([' <=50K', ' <=50K', ' >50K', ' <=50K', ' >50K'], dtype=object)



Podemos conocer cuáles son esos parámetros mirando el atributo `best_params_`.


{{< highlight "python" "linenos=false">}}
model_grid_search.best_params_
{{< /highlight >}}




    {'classifier__learning_rate': 0.1, 'classifier__max_leaf_nodes': 30}



La precisión y los mejores parámetros del pipeline de grid-search son similares a los que encontramos anteriormente, donde localizamos los mejores parámetros "a mano" usando un doble bucle for. Además, podemos inspeccionar todos los resultados, los cuales se almacenan en el atributo `cv_results_` de grid-search. Filtraremos algunas columnas específicas de estos resultados.


{{< highlight "python" "linenos=false">}}
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()
{{< /highlight >}}




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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_classifier__learning_rate</th>
      <th>param_classifier__max_leaf_nodes</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1.032387</td>
      <td>4.768372e-07</td>
      <td>0.214684</td>
      <td>0.006005</td>
      <td>0.1</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.867766</td>
      <td>0.867649</td>
      <td>0.867708</td>
      <td>0.000058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.762404</td>
      <td>4.478788e-02</td>
      <td>0.288248</td>
      <td>0.020517</td>
      <td>0.1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.866729</td>
      <td>0.866557</td>
      <td>0.866643</td>
      <td>0.000086</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.374071</td>
      <td>4.829156e-02</td>
      <td>0.169396</td>
      <td>0.021768</td>
      <td>1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.860559</td>
      <td>0.861261</td>
      <td>0.860910</td>
      <td>0.000351</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.241707</td>
      <td>8.007050e-03</td>
      <td>0.133865</td>
      <td>0.009759</td>
      <td>1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.857993</td>
      <td>0.861862</td>
      <td>0.859927</td>
      <td>0.001934</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.545218</td>
      <td>5.529642e-02</td>
      <td>0.250966</td>
      <td>0.017766</td>
      <td>0.1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.852752</td>
      <td>0.854272</td>
      <td>0.853512</td>
      <td>0.000760</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Con solo dos parámetros podriamos visualizar el grid-search con un mapa de calor. Necesitamos transformar nuestro `cv_results` en un dataframe donde:

+ las filas corresponderán a los valores de la tasa de aprendizaje;
+ las columnas corresponderán al mnúmero máximo de hojas;
+ el contenido del dataframe serán las puntuaciones de prueba medias.


{{< highlight "python" "linenos=false">}}
pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score", index=['param_classifier__learning_rate'],
    columns=["param_classifier__max_leaf_nodes"])
pivoted_cv_results
{{< /highlight >}}




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
      <th>param_classifier__max_leaf_nodes</th>
      <th>3</th>
      <th>10</th>
      <th>30</th>
    </tr>
    <tr>
      <th>param_classifier__learning_rate</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.01</th>
      <td>0.797166</td>
      <td>0.817832</td>
      <td>0.845541</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>0.853512</td>
      <td>0.866643</td>
      <td>0.867708</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>0.860910</td>
      <td>0.859927</td>
      <td>0.851547</td>
    </tr>
    <tr>
      <th>10.00</th>
      <td>0.283476</td>
      <td>0.618080</td>
      <td>0.351642</td>
    </tr>
  </tbody>
</table>
</div>



Podemos usa una representación de mapa de calor para mostrar visualmente el dataframe anterior.


{{< highlight "python" "linenos=false">}}
import seaborn as sns

ax = sns.heatmap(pivoted_cv_results, annot=True, cmap="YlGnBu", vmin=0.7,
                 vmax=0.9)
ax.invert_yaxis()
{{< /highlight >}}


    
![png](/images/output_46_0.png)
    


Observando el mapa de calor podemos resaltar algunas cosas:

+ Para valores muy altos de `learning_rate`, el rendimiento de generalización del modelo se degrada y ajustar el valor de `max_leaf_nodes` no arregla el problema;
+ fuera de esta región problemática, observamos que la opción óptima de `max_leaf_nodes` depende del valor de `learning_rate`;
+ en particular, observamos una "diagonal" de buenos modelos con una precisión cercana al máximo de 0.87: cuando el valor de `max_leaf_nodes` se incrementa, debemos disminuir el valor de `learning_rate` acordemente para mantener una buena precisión.

Por ahora, tengamos en cuenta que, en general, **no existe una única configuración óptima de parámetros**: 4 modelos de las 12 configuraciones de parámetros alcanzan la máxima precisión (hasta pequeñas fluctuaciones aleatorias causadas por el muestreo del conjunto de entrenamiento).

# Ajuste de hiperparámetros por *randomized-search*

Hemos visto que el enfoque de grid-search tiene sus limitaciones. No escala cuando el número de parámetros a ajustar aumenta. Además, grid-search impone una regularidad durante la búsqueda que podría ser problemática. Vamos a presentar otro método para ajustar hiperparámetros denominado búsqueda aleatoria.

Partimos del mismo dataset, el cual hemos dividido en entrenamiento y prueba, y hemos realizado el mismo pipeline de preprocesado.


{{< highlight "python" "linenos=false">}}
model
{{< /highlight >}}




<style>#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a {color: black;background-color: white;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a pre{padding: 0;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-toggleable {background-color: white;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-estimator:hover {background-color: #d4ebff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-item {z-index: 1;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-parallel-item:only-child::after {width: 0;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a div.sk-text-repr-fallback {display: none;}</style><div id="sk-1fa48bfd-9d1b-43f1-b949-c6f9c13aff1a" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7e48de5e-8168-4835-b69c-9ec24673ba5e" type="checkbox" ><label for="7e48de5e-8168-4835-b69c-9ec24673ba5e" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fb66a536-f88c-46e9-9efe-dc3d9686d310" type="checkbox" ><label for="fb66a536-f88c-46e9-9efe-dc3d9686d310" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ae3c149c-f963-4280-a7dd-9caca9c7dbd8" type="checkbox" ><label for="ae3c149c-f963-4280-a7dd-9caca9c7dbd8" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c1714508-41f1-41e2-a3f3-0eb8bb34417d" type="checkbox" ><label for="c1714508-41f1-41e2-a3f3-0eb8bb34417d" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d48a6f83-055d-496b-86cd-54d0851a5179" type="checkbox" ><label for="d48a6f83-055d-496b-86cd-54d0851a5179" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dd040672-ebbe-4422-9bb5-d6a4c819862c" type="checkbox" ><label for="dd040672-ebbe-4422-9bb5-d6a4c819862c" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5da80bc8-f247-45f3-8c48-9b392bf77693" type="checkbox" ><label for="5da80bc8-f247-45f3-8c48-9b392bf77693" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div>



Con el estimador `GridSearchCV`, los parámetros necesitan ser explicitamente especificados. Ya mencionamos que explorar un gran número de valores para diferentes valores sería rápidamente intratable. En su lugar, podemos generar aleatoriamente parámetros candidatos. De hecho, este enfoque evita la regularidad de grid-search. Por tanto, agregar más evaluaciones puede aumentar la resolución en cada dirección. Este es el caso de la frecuente situación en la que la elección de algunos hiperparámetros no es muy importante, como ocurre con el hiperparámetro 2 del siguiente diagrama.

![](/images/randomized_search.png)

De hecho, el número de puntos de evaluación debe ser dividido entre los dos diferentes hiperparámetros. Con un grid-search, el peligro es que esta región de buenos hiperparámetros quede entre la línea del grid: esta región está alineada con el grid dado que el hiperparámetro 2 tiene una influencia débil. Por contra, la búsqueda estocástica muestreará el hiperparámetro 1 independientemente del hiperparámetro 2 y buscará la región óptima.

La clase `RandomizedSearchCV` permite esta búsqueda estocástica. Se usa de forma similar a `GridSearchCV` pero se necesitan especificar las distribuciones de muestreo en lugar de los valores de los parámetros. Por ejemplo, dibujaremos candidatos usando una distribución logarítmica uniforme porque los parámetros que nos interesan toman valores posivos con una escala logarítmica natural (.1 es tan cercano a 1 como éste lo es a 10).

Normalmente, para optimizar 3 o más hiperparámetros, la búsqueda aleatoria es más beneficiosa que grid-search.

Optimizaremos otros 3 parámetros además de los que ya optimizamos con grid-search:

+ `l2_regularization`: corresponde con la fortaleza de la regularización;
+ `min_samples_leaf`: corresponde con el número mínimo de muestras requerida en una hoja;
+ `max_bins`: corresponde con el número máximo de contenedores para construir histogramas.

Podemos usar `scipy.stats.loguniform` para generar números flotantes. Para generar valores aleatorios para parámetros con valores enteros (por ejemplo, `min_samples_leaf`) podemos adaptarlo como sigue:


{{< highlight "python" "linenos=false">}}
from scipy.stats import loguniform

class loguniform_int:
    """versión para valores enteror de la distribución log-uniform"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Ejemplo de variable aleatoria"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
{{< /highlight >}}

Ahora podemos definir la búsqueda aleatoria usando diferentes distribuciones. Ejecutar 10 iteraciones de 5-particiones de validación cruzada para parametrizaciones aleatorias de este modelo en este dataset puede llevar desde 10 segundos a varios minutos, dependiendo de la velocidad de la máquina y del número de procesadores disponibles.


{{< highlight "python" "linenos=false">}}
%%time
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization': loguniform(1e-6, 1e3),
    'classifier__learning_rate': loguniform(0.001, 10),
    'classifier__max_leaf_nodes': loguniform_int(2, 256),
    'classifier__min_samples_leaf': loguniform_int(1, 100),
    'classifier__max_bins': loguniform_int(2, 255),
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    cv=5, verbose=1, n_jobs=-1
)
model_random_search.fit(X_train, y_train)
{{< /highlight >}}

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    CPU times: total: 4.31 s
    Wall time: 7.23 s
    




<style>#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced {color: black;background-color: white;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced pre{padding: 0;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-toggleable {background-color: white;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-estimator:hover {background-color: #d4ebff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-item {z-index: 1;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-parallel-item:only-child::after {width: 0;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced div.sk-text-repr-fallback {display: none;}</style><div id="sk-5fc71a5d-1746-48f6-a3f4-0b91ba882ced" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,Hi...&#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x0000023201C853F0&gt;,&#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86110&gt;,&#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86020&gt;,&#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C844F0&gt;},verbose=1)</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f79df122-00b0-4716-8860-2d88fb39dd9f" type="checkbox" ><label for="f79df122-00b0-4716-8860-2d88fb39dd9f" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=5,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,Hi...&#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x0000023201C853F0&gt;,&#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86110&gt;,&#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86020&gt;,&#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C844F0&gt;},verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2e9618ad-50fc-4702-8035-01f8100e512e" type="checkbox" ><label for="2e9618ad-50fc-4702-8035-01f8100e512e" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6a84eb36-040c-45c0-b91c-6b412f1a14e6" type="checkbox" ><label for="6a84eb36-040c-45c0-b91c-6b412f1a14e6" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fcd82118-70d1-479a-8db4-95796e9871ba" type="checkbox" ><label for="fcd82118-70d1-479a-8db4-95796e9871ba" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bff29f77-ac91-42f1-86aa-07a124f1548e" type="checkbox" ><label for="bff29f77-ac91-42f1-86aa-07a124f1548e" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="22b2cb47-edd6-4818-8375-3644f9ae8a22" type="checkbox" ><label for="22b2cb47-edd6-4818-8375-3644f9ae8a22" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8894a2b6-365a-475a-9eba-5199ad212a48" type="checkbox" ><label for="8894a2b6-365a-475a-9eba-5199ad212a48" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



Después podemos calcular la puntuación de precisión en el conjunto de prueba.


{{< highlight "python" "linenos=false">}}
accuracy = model_random_search.score(X_test, y_test)
print(f"La puntuación de precisión de prueba del mejor modelo es:"
      f"{accuracy:.2f}")
{{< /highlight >}}

    La puntuación de precisión de prueba del mejor modelo es:0.87
    


{{< highlight "python" "linenos=false">}}
model_random_search.best_params_
{{< /highlight >}}




    {'classifier__l2_regularization': 0.0006474800575651534,
     'classifier__learning_rate': 0.9584980078111938,
     'classifier__max_bins': 131,
     'classifier__max_leaf_nodes': 23,
     'classifier__min_samples_leaf': 98}



Como ya vimos, podemos inspeccionar los resultados usando el atributo `cv_results`.


{{< highlight "python" "linenos=false">}}
cv_results
{{< /highlight >}}




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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_classifier__learning_rate</th>
      <th>param_classifier__max_leaf_nodes</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1.032387</td>
      <td>4.768372e-07</td>
      <td>0.214684</td>
      <td>0.006005</td>
      <td>0.1</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.867766</td>
      <td>0.867649</td>
      <td>0.867708</td>
      <td>0.000058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.762404</td>
      <td>4.478788e-02</td>
      <td>0.288248</td>
      <td>0.020517</td>
      <td>0.1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.866729</td>
      <td>0.866557</td>
      <td>0.866643</td>
      <td>0.000086</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.374071</td>
      <td>4.829156e-02</td>
      <td>0.169396</td>
      <td>0.021768</td>
      <td>1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.860559</td>
      <td>0.861261</td>
      <td>0.860910</td>
      <td>0.000351</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.241707</td>
      <td>8.007050e-03</td>
      <td>0.133865</td>
      <td>0.009759</td>
      <td>1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.857993</td>
      <td>0.861862</td>
      <td>0.859927</td>
      <td>0.001934</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.545218</td>
      <td>5.529642e-02</td>
      <td>0.250966</td>
      <td>0.017766</td>
      <td>0.1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.852752</td>
      <td>0.854272</td>
      <td>0.853512</td>
      <td>0.000760</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.309014</td>
      <td>1.376224e-02</td>
      <td>0.131613</td>
      <td>0.006506</td>
      <td>1</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.849749</td>
      <td>0.853344</td>
      <td>0.851547</td>
      <td>0.001798</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.302117</td>
      <td>5.504763e-02</td>
      <td>0.221190</td>
      <td>0.011009</td>
      <td>0.01</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.01, 'classifie...</td>
      <td>0.843252</td>
      <td>0.847830</td>
      <td>0.845541</td>
      <td>0.002289</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.884009</td>
      <td>4.879224e-02</td>
      <td>0.331785</td>
      <td>0.014512</td>
      <td>0.01</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.01, 'classifie...</td>
      <td>0.818956</td>
      <td>0.816708</td>
      <td>0.817832</td>
      <td>0.001124</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.568239</td>
      <td>5.479753e-02</td>
      <td>0.272483</td>
      <td>0.028774</td>
      <td>0.01</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 0.01, 'classifie...</td>
      <td>0.797882</td>
      <td>0.796451</td>
      <td>0.797166</td>
      <td>0.000715</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.200672</td>
      <td>2.001750e-02</td>
      <td>0.090077</td>
      <td>0.011010</td>
      <td>10</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 10, 'classifier_...</td>
      <td>0.742356</td>
      <td>0.493803</td>
      <td>0.618080</td>
      <td>0.124277</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.177403</td>
      <td>1.376188e-02</td>
      <td>0.084322</td>
      <td>0.009258</td>
      <td>10</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 10, 'classifier_...</td>
      <td>0.364545</td>
      <td>0.338739</td>
      <td>0.351642</td>
      <td>0.012903</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.214183</td>
      <td>5.004406e-03</td>
      <td>0.123356</td>
      <td>0.020768</td>
      <td>10</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 10, 'classifier_...</td>
      <td>0.279701</td>
      <td>0.287251</td>
      <td>0.283476</td>
      <td>0.003775</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



Tengamos en mente que este ajuste está limitado por el número de combinaciones diferentes de parámetros que se puntúan mediante búsqueda aleatoria. De hecho, puede haber otros conjuntos de parámetros que conduzcan a un similar o mejor rendimiento de generalización pero que no hayan sido probados en la búsqueda. En la práctica, la búsqueda aleatoria de hiperparámetros se ejecuta con un gran número de iteraciones. Para evitar el coste computacional y aun así realizar un análisis decente, cargamos los resultados obtenidos de una búsqueda similar con 200 iteraciones.


{{< highlight "python" "linenos=false">}}
%%time
model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=200,
    cv=5, verbose=1, n_jobs=-1
)
model_random_search.fit(X_train, y_train)
{{< /highlight >}}

    Fitting 5 folds for each of 200 candidates, totalling 1000 fits
    CPU times: total: 39.7 s
    Wall time: 1min 28s
    




<style>#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 {color: black;background-color: white;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 pre{padding: 0;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-toggleable {background-color: white;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-item {z-index: 1;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-parallel-item:only-child::after {width: 0;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-14b38d11-fbf7-43ba-a230-21f8dfbb7cc5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,Hi...&#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x0000023201C853F0&gt;,&#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86110&gt;,&#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86020&gt;,&#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C844F0&gt;},verbose=1)</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="26927545-9ed9-4507-8d82-ca84908bca10" type="checkbox" ><label for="26927545-9ed9-4507-8d82-ca84908bca10" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=5,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,Hi...&#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x0000023201C853F0&gt;,&#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86110&gt;,&#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C86020&gt;,&#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x0000023201C844F0&gt;},verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="26cc7c2c-7391-4299-a16b-1884c300ba7d" type="checkbox" ><label for="26cc7c2c-7391-4299-a16b-1884c300ba7d" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="44bcf83a-b6b5-43a0-8509-f4f20ecb8d9f" type="checkbox" ><label for="44bcf83a-b6b5-43a0-8509-f4f20ecb8d9f" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e95e41a4-20eb-47c0-b782-dbf84244a0d4" type="checkbox" ><label for="e95e41a4-20eb-47c0-b782-dbf84244a0d4" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f9a376fe-a745-448b-b194-02528f5729fc" type="checkbox" ><label for="f9a376fe-a745-448b-b194-02528f5729fc" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cbc62a39-d65d-4fb1-9741-2b42a89e932f" type="checkbox" ><label for="cbc62a39-d65d-4fb1-9741-2b42a89e932f" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="56fcfad3-d61d-4aa8-aea7-b74d4a9ed5a2" type="checkbox" ><label for="56fcfad3-d61d-4aa8-aea7-b74d4a9ed5a2" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>




{{< highlight "python" "linenos=false">}}
accuracy = model_random_search.score(X_test, y_test)
print(f"La puntuación de precisión de prueba del mejor modelo es: "
      f"{accuracy:.2f}")
{{< /highlight >}}

    La puntuación de precisión de prueba del mejor modelo es: 0.88
    


{{< highlight "python" "linenos=false">}}
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
cv_results = pd.DataFrame(model_random_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)

cv_results = cv_results.rename(shorten_param, axis=1)
cv_results.head()
{{< /highlight >}}




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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>l2_regularization</th>
      <th>learning_rate</th>
      <th>max_bins</th>
      <th>max_leaf_nodes</th>
      <th>min_samples_leaf</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>1.229856</td>
      <td>0.098360</td>
      <td>0.121906</td>
      <td>0.002678</td>
      <td>0.947496</td>
      <td>0.126528</td>
      <td>173</td>
      <td>21</td>
      <td>3</td>
      <td>{'classifier__l2_regularization': 0.9474964020...</td>
      <td>0.868705</td>
      <td>0.872782</td>
      <td>0.873191</td>
      <td>0.866912</td>
      <td>0.870325</td>
      <td>0.870383</td>
      <td>0.002388</td>
      <td>1</td>
    </tr>
    <tr>
      <th>186</th>
      <td>1.809153</td>
      <td>0.109481</td>
      <td>0.156434</td>
      <td>0.003111</td>
      <td>6.955393</td>
      <td>0.136036</td>
      <td>168</td>
      <td>32</td>
      <td>1</td>
      <td>{'classifier__l2_regularization': 6.9553925685...</td>
      <td>0.867886</td>
      <td>0.872236</td>
      <td>0.871007</td>
      <td>0.867731</td>
      <td>0.871690</td>
      <td>0.870110</td>
      <td>0.001920</td>
      <td>2</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1.138879</td>
      <td>0.130649</td>
      <td>0.128410</td>
      <td>0.010519</td>
      <td>16.334498</td>
      <td>0.25793</td>
      <td>252</td>
      <td>16</td>
      <td>4</td>
      <td>{'classifier__l2_regularization': 16.334498052...</td>
      <td>0.868978</td>
      <td>0.872372</td>
      <td>0.870461</td>
      <td>0.864728</td>
      <td>0.871963</td>
      <td>0.869701</td>
      <td>0.002760</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1.520606</td>
      <td>0.067083</td>
      <td>0.144124</td>
      <td>0.011037</td>
      <td>0.253969</td>
      <td>0.040045</td>
      <td>138</td>
      <td>28</td>
      <td>8</td>
      <td>{'classifier__l2_regularization': 0.2539688800...</td>
      <td>0.866112</td>
      <td>0.871553</td>
      <td>0.869915</td>
      <td>0.865684</td>
      <td>0.868960</td>
      <td>0.868445</td>
      <td>0.002243</td>
      <td>4</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.409552</td>
      <td>0.040553</td>
      <td>0.064756</td>
      <td>0.010694</td>
      <td>0.000011</td>
      <td>0.955962</td>
      <td>222</td>
      <td>5</td>
      <td>4</td>
      <td>{'classifier__l2_regularization': 1.1497260106...</td>
      <td>0.860516</td>
      <td>0.870598</td>
      <td>0.868823</td>
      <td>0.866230</td>
      <td>0.864182</td>
      <td>0.866070</td>
      <td>0.003536</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Como tenemos más de 2 parámetros en nuestra búsqueda aleatoria no podemos visualizar los resultados con un mapa de calor. Aún podríamos hacerlo por parejas, pero tener una proyección bidimensional de un problema multidimensional nos puede conducir a una interpresación errónea de las puntuaciones.


{{< highlight "python" "linenos=false">}}
import numpy as np

df = pd.DataFrame(
    {
        "max_leaf_nodes": cv_results["max_leaf_nodes"],
        "learning_rate": cv_results["learning_rate"],
        "score_bin": pd.cut(
            cv_results["mean_test_score"], bins=np.linspace(0.5, 1.0, 6)
        ),
    }
)
sns.set_palette("YlGnBu_r")
ax = sns.scatterplot(
    data=df,
    x="max_leaf_nodes",
    y="learning_rate",
    hue="score_bin",
    s=50,
    color="k",
    edgecolor=None,
)
ax.set_xscale("log")
ax.set_yscale("log")

_ = ax.legend(title="mean_test_score", loc="center left", bbox_to_anchor=(1, 0.5))
{{< /highlight >}}


    
![png](/images/output_67_0.png)
    


En el gráfico podemos ver que las mejores ejecuciones se encuentran en un rango de tasa de aprendizaje de entre 0.01 y 1.0, pero no tenemos control sobre cómo interactúan los otros hiperparámetros en la tasa de aprendizaje. En su lugar, podemos visualizar todos los hiperparámetros al mismo tiempo usando un gráfico de coordenadas paralelas.


{{< highlight "python" "linenos=false">}}
cv_results["l2_regularization"] = cv_results["l2_regularization"].astype("float64")
cv_results["learning_rate"] = cv_results["learning_rate"].astype("float64")
cv_results["max_bins"] = cv_results["max_bins"].astype("float64")
cv_results["max_leaf_nodes"] = cv_results["max_leaf_nodes"].astype("float64")
cv_results["min_samples_leaf"] = cv_results["min_samples_leaf"].astype("float64")
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
import plotly.express as px

fig = px.parallel_coordinates(
    cv_results.rename(shorten_param, axis=1).apply(
        {
            "learning_rate": np.log10,
            "max_leaf_nodes": np.log2,
            "max_bins": np.log2,
            "min_samples_leaf": np.log10,
            "l2_regularization": np.log10,
            "mean_test_score": lambda x: x,
        }
    ),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()
{{< /highlight >}}


![](/images/parallel_coordinates.png)


Transformamos la mayoría de los valores de los ejes tomando log10 o log2 para distribuir los rangos activos y mejorar la legibilidad del gráfico.

El gráfico de coordenadas paralelas muestra los valores de los hiperparámetros en diferentes columnas, mientras que la métrica de rendimiento está codificada por colores. Por tanto, somos capaces de inspeccionar rápidamente si existe un rango de hiperparámetros que funcionan o no.

Es posible **seleccionar un rango de resultados haciendo clic y manteniendo presionado cualquier eje** de coordenadas paralelas del gráfico. Luego podemos deslizar (mover) la selección del rango y cruzar dos selecciones para ver las interacciones. Podemos deshacer la selección haciendo clic una vez más en el mismo eje.

En particular para esta búsqueda de hiperparámetros, es interesante confirmar que las líneas amarillas (modelos de mejor rendimiento) alcanzan valores intermedios para la tasa de aprendizaje, es decir, valores entre las marcas -2 y 0 que corresponden a valores de tasa de aprendizaje de 0,01 y 1, una vez revertimos la transformación log10 para ese eje.

Pero ahora también podemos observar que no es posible seleccionar modelos de mayor rendimiento seleccionado líneas en el eje `max_bins` con valores de marcas entre 1 y 3.

Los otros hiperparámetros no son muy sensibles. Podemos comprobar que si seleccionamos en el eje `learning_rate` valores entre las marcas -1.5 y -0.5 y en el eje `max_bins` valores entre las marcas 5 y 8, siempre seleccionamos modelos con el mejor rendimiento, independientemente de los valores de los otros hiperparámetros.

# Evaluación y ajuste de hiperparámetros

Hasta el momento hemosvisto dos enfoques para ajustar hiperparámetros. Sin embargo, no hemos presentado una forma apropiada para evaluar los modelos "tuneados". En su lugar, nos hemos enfocado en el mecanismo usado para encontrar el mejor conjunto de hiperparámetros. Vamos a mostrar cómo evaluar modelos donde los hiperparámetros necesitan ser ajustados.

Partimos del mismo dataset, el cual hemos dividido en entrenamiento y prueba, y hemos realizado el mismo pipeline de preprocesado.


{{< highlight "python" "linenos=false">}}
model
{{< /highlight >}}




<style>#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 {color: black;background-color: white;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 pre{padding: 0;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-toggleable {background-color: white;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-estimator:hover {background-color: #d4ebff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-item {z-index: 1;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-parallel-item:only-child::after {width: 0;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38 div.sk-text-repr-fallback {display: none;}</style><div id="sk-54bb3de2-7cb5-486a-8f57-b1d5847e7c38" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a554dc3e-8327-4393-aff3-676c3eed6188" type="checkbox" ><label for="a554dc3e-8327-4393-aff3-676c3eed6188" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;,&#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="629b85b4-69bf-424e-93cf-dd00f03f62cc" type="checkbox" ><label for="629b85b4-69bf-424e-93cf-dd00f03f62cc" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b2b496b6-9d1e-4c7d-9e67-e12ffd778df3" type="checkbox" ><label for="b2b496b6-9d1e-4c7d-9e67-e12ffd778df3" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c916ee38-c437-4019-a4e4-5c37c56de886" type="checkbox" ><label for="c916ee38-c437-4019-a4e4-5c37c56de886" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bee4ee51-3a02-4109-be6d-ce36fd35957d" type="checkbox" ><label for="bee4ee51-3a02-4109-be6d-ce36fd35957d" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c4c1e31e-9030-4c8a-9aa8-41c8552589c8" type="checkbox" ><label for="c4c1e31e-9030-4c8a-9aa8-41c8552589c8" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="578b75c1-e1a5-495b-b081-6186d59e65e1" type="checkbox" ><label for="578b75c1-e1a5-495b-b081-6186d59e65e1" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div>



## Evaluación sin ajuste de hiperparámetros


{{< highlight "python" "linenos=false">}}
cv_results = cross_validate(model, X, y, cv=5)
cv_results = pd.DataFrame(cv_results)
cv_results
{{< /highlight >}}




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
      <td>0.527953</td>
      <td>0.057048</td>
      <td>0.863036</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.525952</td>
      <td>0.051044</td>
      <td>0.860784</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.545971</td>
      <td>0.056046</td>
      <td>0.860360</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491422</td>
      <td>0.052546</td>
      <td>0.863124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.495926</td>
      <td>0.048542</td>
      <td>0.867219</td>
    </tr>
  </tbody>
</table>
</div>



Las puntuaciones de validación cruzada provienen de 5-particiones. Entonces, podemos calcular la media y la desviación típica de la puntuación de generalización.


{{< highlight "python" "linenos=false">}}
print(
    f"Puntuación de generalización sin ajuste de hiperparámetros:\n"
    f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}"
)
{{< /highlight >}}

    Puntuación de generalización sin ajuste de hiperparámetros:
    0.863 +/- 0.003
    

## Evaluación con ajuste de hiperparámetros

Vamos a presentar cómo evaluar el modelo con ajuste de hiperparámetros, lo que requiere un paso extra para seleccionar el mejor conjunto de parámetros. Ya vimos que podemos usar una estrategia de búsqueda que utiliza validación cruzada para encontrar el mejor conjunto de hiperparámetros. Aquí vamos a usar una estrategia de grid-search y reproduciremos los pasos que ya vimos anteriormente.

En primer lugar, vamos a incrustar nuestro modelo en un grid-search y especificar los parámetros y los valores de los parámetros que queremos explorar.


{{< highlight "python" "linenos=false">}}
param_grid = {
    'classifier__learning_rate': (0.05, 0.5),
    'classifier__max_leaf_nodes': (10, 30),
}
model_grid_search = GridSearchCV(
    model, param_grid=param_grid, n_jobs=-1, cv=2
)
model_grid_search.fit(X, y)
{{< /highlight >}}




<style>#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c {color: black;background-color: white;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c pre{padding: 0;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-toggleable {background-color: white;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-estimator:hover {background-color: #d4ebff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-item {z-index: 1;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-parallel-item:only-child::after {width: 0;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-d91f18ac-4f33-4d2a-8157-7265844aee0c div.sk-text-repr-fallback {display: none;}</style><div id="sk-d91f18ac-4f33-4d2a-8157-7265844aee0c" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=2,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))]),n_jobs=-1,param_grid={&#x27;classifier__learning_rate&#x27;: (0.05, 0.5),&#x27;classifier__max_leaf_nodes&#x27;: (10, 30)})</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="756e9d3e-1d44-4c6a-9165-ff2339417a8e" type="checkbox" ><label for="756e9d3e-1d44-4c6a-9165-ff2339417a8e" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=2,estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(remainder=&#x27;passthrough&#x27;,sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;,&#x27;education&#x27;,&#x27;marital-status&#x27;,&#x27;occupation&#x27;,&#x27;relationship&#x27;,&#x27;race&#x27;,&#x27;sex&#x27;,&#x27;native-country&#x27;])])),(&#x27;classifier&#x27;,HistGradientBoostingClassifier(max_leaf_nodes=4,random_state=42))]),n_jobs=-1,param_grid={&#x27;classifier__learning_rate&#x27;: (0.05, 0.5),&#x27;classifier__max_leaf_nodes&#x27;: (10, 30)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="55b15c9c-916b-4063-a30b-1128e768e8f2" type="checkbox" ><label for="55b15c9c-916b-4063-a30b-1128e768e8f2" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,transformers=[(&#x27;cat_preprocessor&#x27;,OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,unknown_value=-1),[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,&#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,&#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8419c484-ab91-4477-9a3b-58f8ab96b4c7" type="checkbox" ><label for="8419c484-ab91-4477-9a3b-58f8ab96b4c7" class="sk-toggleable__label sk-toggleable__label-arrow">cat_preprocessor</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bee9b2a6-5cc8-4f64-9588-68d45d894bcf" type="checkbox" ><label for="bee9b2a6-5cc8-4f64-9588-68d45d894bcf" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c0ee9c94-e086-4657-9250-e9a548c1dba2" type="checkbox" ><label for="c0ee9c94-e086-4657-9250-e9a548c1dba2" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d36cd104-a615-43d9-8020-f201908cb0bd" type="checkbox" ><label for="d36cd104-a615-43d9-8020-f201908cb0bd" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9eed132e-c726-4274-93aa-6654ce99c4f8" type="checkbox" ><label for="9eed132e-c726-4274-93aa-6654ce99c4f8" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



Como vimos, cuando llamamos al método `fit`, el modelo embebido en grid-search es entrenado con cada una de las posibles combinaciones de parámetros resultado del cuadrante de parámetros. Se selecciona la mejor combinación, manteniendo aquella combinación que conduce a la mejor puntuación media de validación cruzada.


{{< highlight "python" "linenos=false">}}
cv_results = pd.DataFrame(model_grid_search.cv_results_)
cv_results
{{< /highlight >}}




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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_classifier__learning_rate</th>
      <th>param_classifier__max_leaf_nodes</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.603518</td>
      <td>0.013511</td>
      <td>0.260224</td>
      <td>0.006005</td>
      <td>0.05</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.05, 'classifie...</td>
      <td>0.863970</td>
      <td>0.864707</td>
      <td>0.864338</td>
      <td>0.000369</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.898772</td>
      <td>0.002502</td>
      <td>0.319024</td>
      <td>0.004754</td>
      <td>0.05</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.05, 'classifie...</td>
      <td>0.871013</td>
      <td>0.870317</td>
      <td>0.870665</td>
      <td>0.000348</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.300508</td>
      <td>0.036280</td>
      <td>0.163140</td>
      <td>0.023520</td>
      <td>0.5</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.5, 'classifier...</td>
      <td>0.866426</td>
      <td>0.868679</td>
      <td>0.867553</td>
      <td>0.001126</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.261725</td>
      <td>0.007006</td>
      <td>0.152131</td>
      <td>0.001501</td>
      <td>0.5</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.5, 'classifier...</td>
      <td>0.867164</td>
      <td>0.866836</td>
      <td>0.867000</td>
      <td>0.000164</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




{{< highlight "python" "linenos=false">}}
model_grid_search.best_params_
{{< /highlight >}}




    {'classifier__learning_rate': 0.05, 'classifier__max_leaf_nodes': 30}



Una importante advertencia aquí es la concerniente a la evaluación del rendimiento de generalización. De hecho, la media y la desviación típica de las puntuaciones calculadas por la validación cruzada en grid-search no son potencialmente buenas estimaciones del rendimiento de generalización que obtendríamos reentrenando un modelo con la mejor combinación de valores de hiperparámetros en el dataset completo. Hay que tener en cuenta que scikit-learn, por defecto, ejecuta automáticamente este reentreno cuando llamamos a `model_grid_search.fit`. Este modelo reentrenado se entrena con más datos que los diferentes modelos entrenados internamente durante la validación cruzada de grid-search.

Por lo tanto, usamos el conocimiento del dataset completo para decidir los hiperparámetros de nuestro modelo y entrenar el modelo reajustado. Debido a esto, se debe mantener un conjunto de prueba externo para la evaluación final del modelo reajustado. Destacamos aquí el proceso usando una única división entrenamiento-prueba.


{{< highlight "python" "linenos=false">}}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model_grid_search.fit(X_train, y_train)
accuracy = model_grid_search.score(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy:.3f}")
{{< /highlight >}}

    Precisión en el conjunto de prueba: 0.879
    

La medida de puntuación en el conjunto de prueba final está casi en el mismo rango que la puntuación de validación cruzada interna para la mejor combinación de hiperparámetros. Esto es tranquilizador, ya que significa que el procedimiento de ajuste no ha provocado un overfitting significativo en sí mismo (de lo contrario, la puntuación de prueba final habría sido más baja que la puntuación de validación cruzada interna). Eso era de esperar porque nuestro grid-search exploró muy pocas combinaciones de hiperparámetros en aras de la velocidad. La puntuación de prueba del modelo final es realmente un poco más alta de la que cabría esperar de la validación cruzada interna. Esto también era de esperar porque el modelo reajustado se entrena en un dataset más grande que los modelos evaluados en el bucle de validación cruzada interno del procedimiento de grid-search. Este suele ser el caso de los modelos entrenados con un gran número de instancias, tienden a generalizar mejor.

En el código anterior, la selección de los mejores hiperparámetros se realizó únicamente en el conjunto de entrenamiento de la división inicial entrenamiento-prueba. Después, evaluamos el rendimiento de generalización de nuestro modelo tuneado en el conjunto de prueba restante. Esto se puede mostrar esquemáticamente en el siguiente diagrama:

![](/images/kfold_cv.png)

Esta figura muestra el caso particular de la estrategia de validación cruzada de **K-particiones** usando `n_splits=5` para dividir el conjunto de entrenamiento proveniente de la división entrenamient-prueba. Para cada división de validación cruzada, el procedimiento entrena un modelo en todas las instancias rojas, evalúa la puntuación de un conjunto dado de hiperparámetros en las instancias verdes. Los mejores hiperparámetros se seleccionan basándose en estas puntuaciones intermedias. El modelo final tuneado con esos hiperparámetros se entrena en la concatenación de instancias rojas y verdes y se evalúa en las instancias azules.

Las instancias verdes a menudo se denominan conjuntos de validación para diferenciarlos del conjunto de prueba final en azul.

Sin embargo, esta evaluación solo nos proporciona una estimación puntual única del rendimiento de generalización. Como recordamos al principio, es beneficioso disponer de una idea aproximada de la incertidumbre de nuestro rendimiento de generalización estimado. Por lo tanto, deberíamos usar adicionalmente una validación cruzada para esta estimación.

Este patrón se denomina **validación-cruzada anidada**. Usamos la validación cruzada interna para la selección de los hiperparámetros y la validación cruzada externa para la evaluación del rendimiento de generalización del modelo tuneado reajustado.

En la práctica, solo necesitamos incrustar grid-search en la función `cross-validate` para ejecutar dicha evaluación.


{{< highlight "python" "linenos=false">}}
cv_results = cross_validate(
    model_grid_search, X, y, cv=5, n_jobs=-1, return_estimator=True
)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
cv_results = pd.DataFrame(cv_results)
cv_test_scores = cv_results["test_score"]
print(
    "Puntuación de generalización con ajuste de hiperparámetros:\n"
    f"{cv_test_scores.mean():.3f} +/- {cv_test_scores.std():.3f}"
)
{{< /highlight >}}

    Puntuación de generalización con ajuste de hiperparámetros:
    0.871 +/- 0.003
    



Este resultado es compatible con la puntuación de prueba medida en la división externa entrenamiento-prueba. Sin embargo, en este caso obtenemos conocimiento sobre la variablidad de nuestra estimación del rendimiento de generalización gracias a la medida de la desviación típica de las puntuaciones medidas en la validación cruzada externa.

A continuación se muestra una representación esquemática del procedimiento completo de validación cruzada anidada.

![](/images/kf_cv_nested.png)

En la figura se ilustra la estrategia de validación cruzada anidada usando `cv_inner = Kfold(n_splits=4)` y `cv_outer = Kfold(n_splits=5)`.

Para cada división de validación cruzada interna (indexada en la parte izquierda), el procedimiento entrena un modelo en todas las muestras rojas y evalúa la calidad de los hiperparámetros en las muestras verdes.

Para cada división de validación cruzada externa (indexada en la parte derecha), se seleccionan los mejores hiperparámetros basándose en las puntuaciones de validación (calculadas en las muestras verdes) y se reajusta un modelo en la concatenación de las instancias rojas y verdes para esa iteración de validación cruzada externa.

El rendimiento de generalización de los 5 modelos reajustados del bucle de validación cruzada externa se evalúa en las instancias azules para obtener las puntuaciones finales.

Pasando el parámetro `return_estimator=True` podemos comprobar el valor de los mejores hiperparámetros obtenidos para cada partición de la validación cruzada externa.


{{< highlight "python" "linenos=false">}}
for cv_fold, estimator_in_fold in enumerate(cv_results["estimator"]):
    print(
        f"Mejores hiperparámetros para la partición nº{cv_fold+1}:\n"
        f"{estimator_in_fold.best_params_}"
    )
{{< /highlight >}}

    Mejores hiperparámetros para la partición nº1:
    {'classifier__learning_rate': 0.05, 'classifier__max_leaf_nodes': 30}
    Mejores hiperparámetros para la partición nº2:
    {'classifier__learning_rate': 0.05, 'classifier__max_leaf_nodes': 30}
    Mejores hiperparámetros para la partición nº3:
    {'classifier__learning_rate': 0.05, 'classifier__max_leaf_nodes': 30}
    Mejores hiperparámetros para la partición nº4:
    {'classifier__learning_rate': 0.5, 'classifier__max_leaf_nodes': 10}
    Mejores hiperparámetros para la partición nº5:
    {'classifier__learning_rate': 0.05, 'classifier__max_leaf_nodes': 30}
    

Es interesante ver si el procedimiento de ajuste de hiperparámetros siempre selecciona valores similares para los hiperparámetros. Si es el caso, entonces todo está bien. Significa que podemos desplegar un modelo ajustado con esos hiperparámetros y esperar que tenga un rendimiento predictivo real cercano al que medimos en la validación cruzada externa.

Pero también es posible que algunos hiperparámetros no tengan ninguna importancia y, como resultado de diferentes sesiones de ajuste, den resultados diferentes. En este caso, servirá cualquier valor. Normalmente esto se puede confirmar haciendo un gráfico de coordenadas paralelas de los resultados de una gran búsqueda de hiperparáemtros, como ya vimos.

Desde el punto de vista de la implementación, se podría optar por implementar todos los modelos encontrados en el ciclo de validación cruzada externa y votar para obtener las predicciones finales. Sin embargo, esto puede causar problemas operativos debido a que usa más memoria y hace que la predicción sea más lenta, lo que resulta en un mayor uso de recursos computacionales por predicción.

# Resumen

+ Los hiperparámetros tienen un impacto en el rendimiento de los modelos y deben ser elegirse sabiamente;
+ La búsqueda de los mejores hiperparámetros se puede automatizar con un enfoque de grid-search o búsqueda automática;
+ Grid-search es costoso y no escala cuando el número de hiperparámetros a optimizar incrementa. Además, la combinación se muestrea únicamente en una retícula regular.
+ Una búsqueda aleatoria permite buscar con una propuesta fija incluso con un número creciente de hiperparámetros. Además, la combinación se muestrea en una retícula no regular.

+ El **overfitting** es causado por el tamaño limitado del conjunto de entrenamiento, el ruido en los datos y la alta flexibilidad de los modelos de machine learning comunes.

+ El **underfitting** sucede cuando las funciones de predicción aprendidas sufren de **errores sistemáticos**. Esto se puede producir por la elección de la familia del modelo y los parámetros, lo cuales conducen a una **carencia de flexibilidad** para capturar la estructura repetible del verdadero proceso de generación de datos.

+ Para un conjunto de entrenamiento dado, el objetivo es **minimizar el error de preba** ajustando la familia del modelo y sus parámetros para encontrar el **mejor equilibrio entre overfitting y underfitting**.

+ Para una familia de modelo y parámetros dados, **incrementar el tamaño del conjunto de entrenamiento disminuirá el overfitting**, pero puede causar un incremento del underfitting.

+ El error de prueba de un modelo que no tiene overfitting ni underfitting puede ser alto todavía si las variaciones de la variable objetivo no pueden ser determinadas completamente por las variables de entrada. Este error irreductible es causado por lo que algunas veces llamamos error de etiqueta. En la práctica, esto sucede a menudo cuando por una razón u otra no tenemos acceso a features importantes.

Algunas referencias a seguir con ejemplos de algunos conceptos mencionados:

+ [Ejemplo de un grid-search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py)
+ [Ejemplo de una búsqueda aleatoria](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py)
+ [Ejemplo de una validación cruzada anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)
