---
title: "Pipeline de modelado predictivo"
date: 2022-03-07T18:04:46+01:00
tags: [pipeline, regresión]
categories: [tutoriales]
---

En este post vamos a presentar un ejemplo de un pipeline de modelado predictivo típico usando datos tabulares, es decir, que pueden ser estructurados en una tabla de 2 dimensiones. En primer lugar, analizaremos el dataset usado y posteriormente entrenaremos un primer pipeline predictivo. Después prestaremos atención a los tipos de datos que tiene que manejar nuestro modelo: numéricos y categóricos. Por último, extenderemos nuestro pipeline para tipos de datos mixtos, es decir, numéricos y categóricos.

El objetivo a conseguir es construir intuiciones respecto a un dataset desconocido, identificar y discriminar *features* numéricas y categóricas y, finalmente, crear un pipeline predictivo avanzado con *scikit-learn*.

# Primer vistazo al dataset

Antes de llevar a cabo cualquier tarea de machine learning hay que realizar un serie de pasos:

+ cargar los datos.
+ observar las variables del dataset, diferenciando entre variables numéricas y categóricas, las cuales necesitarán un preprocesamiento diferente en la mayoría de los flujos de machine learning.
+ visualizar la distribución de las variables para obtener algún tipo de conocimiento o idea del dataset.

Usaremos el dataset "**credit-g**". Para más detalles sobre dicho dataset puedes acceder al link [https://www.openml.org/d/31](https://www.openml.org/d/31). El objetivo del dataset es clasificar a las personas por un conjunto de atributos como buenas o malas respecto al riesgo crediticio. Los datos están disponibles en un fichero CSV y usaremos pandas para leerlo. 


```python
import numpy as np
import pandas as pd

credit = pd.read_csv("credit-g.csv")
```

## Las variables del dataset

Los datos se almacenan en un *dataframe* de pandas. Un dataframe es una estructura de datos de 2 dimensiones. Este tipo de datos también se denominan datos tabulares. 

Cada fila representa un "ejemplo". En el campo de machine learning se usan normalmente los términos equivalentes de "registro", "instancia" u "observación". 

Cada columna representa un tipo de información que ha sido recopilada y se denominan "features". En el campo de machine learning es normal usar los términos equivalentes de "variable", "atributo" o "covariable".

Echemos un vistazo rápido al dataframe para mostrar las primeras filas:


```python
credit.head()
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
      <th>checking_status</th>
      <th>duration</th>
      <th>credit_history</th>
      <th>purpose</th>
      <th>credit_amount</th>
      <th>savings_status</th>
      <th>employment</th>
      <th>installment_commitment</th>
      <th>personal_status</th>
      <th>other_parties</th>
      <th>...</th>
      <th>property_magnitude</th>
      <th>age</th>
      <th>other_payment_plans</th>
      <th>housing</th>
      <th>existing_credits</th>
      <th>job</th>
      <th>num_dependents</th>
      <th>own_telephone</th>
      <th>foreign_worker</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>'&lt;0'</td>
      <td>6</td>
      <td>'critical/other existing credit'</td>
      <td>radio/tv</td>
      <td>1169</td>
      <td>'no known savings'</td>
      <td>'&gt;=7'</td>
      <td>4</td>
      <td>'male single'</td>
      <td>none</td>
      <td>...</td>
      <td>'real estate'</td>
      <td>67</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>yes</td>
      <td>yes</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'0&lt;=X&lt;200'</td>
      <td>48</td>
      <td>'existing paid'</td>
      <td>radio/tv</td>
      <td>5951</td>
      <td>'&lt;100'</td>
      <td>'1&lt;=X&lt;4'</td>
      <td>2</td>
      <td>'female div/dep/mar'</td>
      <td>none</td>
      <td>...</td>
      <td>'real estate'</td>
      <td>22</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>'no checking'</td>
      <td>12</td>
      <td>'critical/other existing credit'</td>
      <td>education</td>
      <td>2096</td>
      <td>'&lt;100'</td>
      <td>'4&lt;=X&lt;7'</td>
      <td>2</td>
      <td>'male single'</td>
      <td>none</td>
      <td>...</td>
      <td>'real estate'</td>
      <td>49</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>'unskilled resident'</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'&lt;0'</td>
      <td>42</td>
      <td>'existing paid'</td>
      <td>furniture/equipment</td>
      <td>7882</td>
      <td>'&lt;100'</td>
      <td>'4&lt;=X&lt;7'</td>
      <td>2</td>
      <td>'male single'</td>
      <td>guarantor</td>
      <td>...</td>
      <td>'life insurance'</td>
      <td>45</td>
      <td>none</td>
      <td>'for free'</td>
      <td>1</td>
      <td>skilled</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>'&lt;0'</td>
      <td>24</td>
      <td>'delayed previously'</td>
      <td>'new car'</td>
      <td>4870</td>
      <td>'&lt;100'</td>
      <td>'1&lt;=X&lt;4'</td>
      <td>3</td>
      <td>'male single'</td>
      <td>none</td>
      <td>...</td>
      <td>'no known property'</td>
      <td>53</td>
      <td>none</td>
      <td>'for free'</td>
      <td>2</td>
      <td>skilled</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
      <td>bad</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
credit.shape
```




    (1000, 21)



El dataset está compuesto de 1.000 instancias y 21 variables. La columna llamada `class` es nuestra variable objetivo (es decir, la variable que queremos predecir). Las dos posibles clases son `good` (bajo riesgo credicitio) y `bad` (alto riesgo crediticio). El problema de predicción resultante es, por tanto, un problema de clasificación binaria. Usaremos el resto de columnas como variables de entrada para nuestro modelo.


```python
credit["class"].value_counts()
```




    good    700
    bad     300
    Name: class, dtype: int64




```python
credit["class"].value_counts().plot.pie(autopct='%1.2f%%');
```


    
![png](/images/output_11_0.png)
    


Vemos que las clases están desbalanceadas, lo que significa que tenemos más instancias de una o más clases comparada con las otras. El desequilibro de clases sucede frecuentemente en la práctica y puede requerir de técnicas especiales al construir el modelo predictivo. Veremos este tipo de técnicas en otros posts.


```python
credit.dtypes
```




    checking_status           object
    duration                   int64
    credit_history            object
    purpose                   object
    credit_amount              int64
    savings_status            object
    employment                object
    installment_commitment     int64
    personal_status           object
    other_parties             object
    residence_since            int64
    property_magnitude        object
    age                        int64
    other_payment_plans       object
    housing                   object
    existing_credits           int64
    job                       object
    num_dependents             int64
    own_telephone             object
    foreign_worker            object
    class                     object
    dtype: object




```python
credit.dtypes.value_counts()
```




    object    14
    int64      7
    dtype: int64



Comprobamos que el dataset contiene tanto datos numéricos (7 features) como categóricos (14 features, incluyendo la variable objetivo). En este caso sus tipos son `int64` y `object`, respectivamente.

## Inspección visual de los datos

Antes de construir cualquier modelo predictivo es buena idea echar un vistazo a los datos:

+ quizás la tarea que estamos intentando conseguir se pueda resolver sin utilizar machine learning;
+ debemos comprobar que la información que necesitamos se encuentra presente realmente en el dataset;
+ inspeccionar los datos en una buena forma de encontrar peculiaridades. Estas pueden aparecer durante la recolección de los datos (por ejemplo, debido al malfuncionamiento de sensores o valores faltantes) o en la forma en que los datos son procesados posteriormente (por ejemplo, valores "capados").

Echemos un vistazo a las distribuciones de las features individualmente para obtener algún conocimiento adicional sobre los datos. Podemos empezar dibujando histogramas, aunque esto solo aplicaría a las features numéricas:


```python
_ = credit.hist(figsize=(20, 14))
```

    C:\Program Files\Python39\lib\site-packages\pandas\plotting\_matplotlib\tools.py:400: MatplotlibDeprecationWarning: 
    The is_first_col function was deprecated in Matplotlib 3.4 and will be removed two minor releases later. Use ax.get_subplotspec().is_first_col() instead.
      if ax.is_first_col():
    


    
![png](/images/output_19_1.png)
    


Algunos comentarios sobre estas variables:

+ `duration`: la mayoría de las personas a las que se les concede el crédito su duración está entre aproximadamente 4 y 24 meses, principalmente entre 12 y 24 meses.
+ `credit_amount`: la mayoría de las personas solicita un crédito menor de 4.000 aproximadamente.
+ `age`: la mayoría de las personas que solicitan un crédito son menores de 40 años.


Veamos la distribución de algunas variables categóricas:


```python
import matplotlib.pyplot as plt
import seaborn as sns

_ = sns.countplot(x="checking_status", data=credit)
```


    
![png](/images/output_22_0.png)
    



```python
ax = sns.countplot(x="credit_history", data=credit)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
```


    
![png](/images/output_23_0.png)
    


Bueno, hasta ahora hemos visto cómo cargar un dataset, calcular su tamaño y visualizar de forma rápida las primeras filas del mismo. En un primer análisis de las variables que lo componen, hemos identificado nuestra variable objetivo y diferenciado las variables numéricas y categóricas. También hemos podido observar cómo se distribuyen sus valores.

# Modelo simple con scikit-learn

Vamos a crear un primer modelo predictivo, para lo cual solo usaremos las variables numéricas. Los datos numéricos son el tipo de datos más natural en machine learning y (casi) pueden incorporarse directamente a los modelos predictivos.

Como hemos visto, el archivo CSV contiene toda la información que necesitamos: el objetivo que nos gustaría predecir (es decir, `class`) y los datos que queremos usar para entrenar nuestro modelo predictivo (es decir, las columnas restantes). El primer paso es separar las columnas para obtener de un lado el objetivo y del otro lado los datos.

## Separar los datos y el objetivo


```python
target_name = "class"
y = credit[target_name]
data = credit.drop(columns=[target_name])
```

Vamos a usar una función de sklearn que nos permite seleccionar las columnas en función del tipo de dato.


```python
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_include=np.number)
numerical_columns = numerical_columns_selector(data)
numerical_columns
```




    ['duration',
     'credit_amount',
     'installment_commitment',
     'residence_since',
     'age',
     'existing_credits',
     'num_dependents']




```python
X = data[numerical_columns]
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
      <th>duration</th>
      <th>credit_amount</th>
      <th>installment_commitment</th>
      <th>residence_since</th>
      <th>age</th>
      <th>existing_credits</th>
      <th>num_dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>1169</td>
      <td>4</td>
      <td>4</td>
      <td>67</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
      <td>5951</td>
      <td>2</td>
      <td>2</td>
      <td>22</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>2096</td>
      <td>2</td>
      <td>3</td>
      <td>49</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>7882</td>
      <td>2</td>
      <td>4</td>
      <td>45</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>4870</td>
      <td>3</td>
      <td>4</td>
      <td>53</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Entrenar un modelo y hacer predicciones

Vamos a construir un modelo de clasificación usando *regresión logística*, que pertenece a la familia de los modelos lineales. 

Brevemente, los modelos lineales buscan un conjunto de pesos para combinar linealmente las features y predecir el objetivo. Por ejemplo, el modelo puede generar un regla como la siguiente:

+ si `0.1 * duration + 3.3 * credit_amount - 15.1 * installment_commitment + 3.2 * residence_since - 0.2 * age + 1.3 * existing_credits - 0.9 * num_dependents + 13.2 > 0`, predice `good`

+ en caso contrario predice `bad`

El metodo `fit` se llama para entrenar el modelo a partir de los datos de entrada (features) y objetivo.


```python
from sklearn import set_config
set_config(display="diagram")
```


```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X, y)
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="25c32ba9-0dc0-4383-9b79-c2b3b7af3001" type="checkbox" checked><label class="sk-toggleable__label" for="25c32ba9-0dc0-4383-9b79-c2b3b7af3001">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500)</pre></div></div></div></div></div>



El proceso de aprendizaje puede representarse de la siguiente forma:

![](images/model_fit.png)

El método `fit`se compone de dos elementos: un algoritmo de aprendizaje y algunos estados del modelo. El algoritmo de aprendizaje toma los datos y el objetivo de entrenamiento como entrada y establece los estados del modelo. Estos estados del modelo se utilizarán posteriormente para predecir (por clasificadores o regresores) o transformar los datos (por transformadores).

Tanto el algoritmo de aprendizaje como el tipo de estados del modelo son específicos para cada tipo de modelo.

Usaremos ahora nuestro modelo para llevar a cabo algunas predicciones usando el mismo dataset.


```python
y_predicted = model.predict(X)
```

El mecanismo de predicción puede representarse de la siguiente forma:

![](images/model_predict.png)

Para predecir, un modelo usa una **función de predicción** que utilizará los datos de entrada junto con los estados del modelo. Como el algoritmo de aprendizaje y los estados del modelo, la función de predicción es específica para cada tipo de modelo.

Vamos a revisar las predicciones calculadas. Por simplicidad vamos a echar un vistazo a los primeros cinco objetivos predichos.


```python
y_predicted[:5]
```




    array(['good', 'bad', 'good', 'good', 'good'], dtype=object)



De hecho, podemos comparar estas predicciones con los datos reales:


```python
y[:5]
```




    0    good
    1     bad
    2    good
    3    good
    4     bad
    Name: class, dtype: object



e incluso podríamos comprobar si las predicciones concuerdan con los objetivos reales:


```python
y_predicted[:5] == y[:5]
```




    0     True
    1     True
    2     True
    3     True
    4    False
    Name: class, dtype: bool




```python
print(f"Nº de predicciones correctas: {(y_predicted[:5] == y[:5]).sum()} de las 5 primeras")
```

    Nº de predicciones correctas: 4 de las 5 primeras
    

En este caso, parece que nuestro modelo comete un error al predecir la quinta instancia. Para obtener un mejor evaluación podemos calcular la tasa promedio de éxito:


```python
(y_predicted == y).mean()
```




    0.706



¿Podemos confiar en esta evaluación? ¿Es buena o mala?

## División de los datos en entrenamiento y prueba

Cuando construimos un modelo de machine learning es muy importante evaluar el modelo entrenado en datos que no se hayan usado para entrenarlo, ya que la **generalización** es más que la memorización (significa que queremos una regla que generalice a nuevos datos, sin comparar los datos memorizados). Es más difícil concluir sobre datos nunca vistos que sobre los ya vistos.

La evaluación correcta se realiza fácilmente reservando un subconjunto de los datos cuando entrenamos el modelo y usándolos posteriormente para evaluar el modelo. Los datos usados para entrenar un modelo se denominan **datos de entrenamiento** mientras que los datos usados para evaluar el modelo se denominan **datos de prueba**.

En ocasiones podemos contar con dos datasets separados, uno para el entrenamiento y otro para pruebas. Sin embargo, esto suele ser bastante inusual. La mayoría de las veces tendremos un único archivo que contiene todos los datos y necesitaremos dividirlo una vez cargado en memoria.

Scikit-learn proporciona la función `sklearn.model_selection.train_test_split`, que usaremos para dividir automáticamente el dataset en dos subconjuntos.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.20)
```

Cuando llamamos a la función `train_test_split`, especificamos que queremos tener el 20% de las instancias en el conjunto de prueba y las instancias restantes (80%) estarán disponibles para el conjunto de entrenamiento.

## Establecimiento de una línea base

Para avaluar el rendimiento de nuestro modelo predictivo resulta de utilidad establecer una línea base simple. La línea base más simple para un clasificador es aquella que predice siempre la misma clase, independientemente de los datos de entrada. Para ello usaremos un `DummyClassifier`.


```python
from sklearn.dummy import DummyClassifier

clf_dummy = DummyClassifier(strategy="most_frequent", random_state=42)
```


```python
clf_dummy.fit(X_train, y_train)
accuracy_dummy = clf_dummy.score(X_test, y_test)
print(f"Accuracy línea base: {accuracy_dummy}")
```

    Accuracy línea base: 0.705
    

Este clasificador dummy predice siempre la clase más frecuente (en nuestro caso, la clase `good`). Como vimos anteriormente la proporción de clase `good` era del 70%, que coincide con la puntuación obtenido por este clasificador. Bien, ya tenemos una linea base con la que comparar nuestro modelo.

Vamos a entrenar el modelo exactamente de la misma forma que vimos anteriormente, excepto que usaremos para ello los subconjuntos de entrenamiento:


```python
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a38317d5-a9cd-4ed5-86de-6059799e45ce" type="checkbox" checked><label class="sk-toggleable__label" for="a38317d5-a9cd-4ed5-86de-6059799e45ce">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500)</pre></div></div></div></div></div>



En lugar de calcular la predicción y calcular manualmente la tasa media de éxito, podemos usar el método `score`. Cuando se trata de clasificadores este método devuelve su métrica de rendimiento.


```python
accuracy_lgr = model.score(X_test, y_test)
print(f"Accuracy: {accuracy_lgr:.3f}")
```

    Accuracy: 0.740
    

Veamos el mecanismo subyacente cuando se llama al método `score`:

![](images/model_score.png)

Para calcular la puntuación, el predictor primero calcula las predicciones (usando el metodo `predict`) y luego usa una función de puntuación para comparar los objetivos reales y las predicciones. Por último, se devuelve la puntuación.

Por norma general, nos referimos al **rendimiento de generalización** de un modelo cuando nos refiramos a la puntuación de prueba o al error de prueba obtenido al comparar la predicción de un modelo con los objetivos reales. También son términos equivalentes rendimiento predictivo y rendimiento estadístico. Nos referimos al **rendimiento computacional** de un modelo predictivo cuando accedemos al coste computacional de entrenar un modelo predictivo o usarlo para hacer predicciones.

Bueno, la puntuación de nuestro modelo apenas mejora la linea base que establecimos:


```python
print(f"Accuracy línea base = {accuracy_dummy}")
print(f"Accuracy regresión logística = {accuracy_lgr}")
```

    Accuracy línea base = 0.705
    Accuracy regresión logística = 0.74
    

Seguro que podemos hacerlo mejor. Veamos cómo.
