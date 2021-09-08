---
title: "30 Days of ML"
date: 2021-09-07T13:51:47+02:00
tags: [kaggle, regresión]
categories: [tutoriales]
draft: false
---

Durante el mes de agosto he participado en el evento organizado por Kaggle denominado ***[30 Days of ML](https://www.kaggle.com/thirty-days-of-ml)***. Las dos primeras semanas consistieron en un repaso a los conceptos básicos de python y machine learning. Las últimas dos semanas participamos en una [competición](https://www.kaggle.com/c/30-days-of-ml) creada para todos los concursantes del evento.

Para la competición disponíamos de una dataset sintético, pero basado en datos reales. El objetivo era predecir la cantidad de una reclamación del seguro. Las *features* estaban anonimizadas, pero relacionadas con features del mundo real. Las columnas de features `cat0` a `cat9` eran categóricas, y las columnas de features `cont0` a `cont13` continuas.

Nos proporcionan los siguientes archivos:

+ **train.csv** - los datos de entrenamiento con la columna target
+ **test.csv** - el conjuto de prueba; tendremos que predecir el target para cada una de las filas de este archivo
+ **sample_submission.csv** - un archivo de envío de ejemplo con el formato correcto

Las semanas previas a la competición, durante el curso de machine learning, trabajamos principalmente con dos modelos: 
+ Random Forest
+ Uso y optimización de modelos con **gradient boosting**. En concreto, hacemos uso de la librería **XGBoost**. 

Por tanto, para esta competición seguí las líneas marcadas durantes las semanas de aprendizaje teórico y utilicé ambos modelos. A continuación detallo los pasos seguidos durante los días que trabajé en la competición.

## Importación de librerías necesarias


{{< highlight "python" "linenos=false">}}
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
{{< /highlight >}}

## Carga de los datasets


{{< highlight "python" "linenos=false">}}
# Carga de los datos de entrenamiento y prueba
train = pd.read_csv("input/train.csv", index_col=0)
test = pd.read_csv("input/test.csv", index_col=0)

# Previsualización del dataset
train.head()
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
      <th>cat0</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>target</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>C</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>E</td>
      <td>C</td>
      <td>N</td>
      <td>...</td>
      <td>0.400361</td>
      <td>0.160266</td>
      <td>0.310921</td>
      <td>0.389470</td>
      <td>0.267559</td>
      <td>0.237281</td>
      <td>0.377873</td>
      <td>0.322401</td>
      <td>0.869850</td>
      <td>8.113634</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>F</td>
      <td>A</td>
      <td>O</td>
      <td>...</td>
      <td>0.533087</td>
      <td>0.558922</td>
      <td>0.516294</td>
      <td>0.594928</td>
      <td>0.341439</td>
      <td>0.906013</td>
      <td>0.921701</td>
      <td>0.261975</td>
      <td>0.465083</td>
      <td>8.481233</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>D</td>
      <td>A</td>
      <td>F</td>
      <td>...</td>
      <td>0.650609</td>
      <td>0.375348</td>
      <td>0.902567</td>
      <td>0.555205</td>
      <td>0.843531</td>
      <td>0.748809</td>
      <td>0.620126</td>
      <td>0.541474</td>
      <td>0.763846</td>
      <td>8.364351</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>E</td>
      <td>C</td>
      <td>K</td>
      <td>...</td>
      <td>0.668980</td>
      <td>0.239061</td>
      <td>0.732948</td>
      <td>0.679618</td>
      <td>0.574844</td>
      <td>0.346010</td>
      <td>0.714610</td>
      <td>0.540150</td>
      <td>0.280682</td>
      <td>8.049253</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>E</td>
      <td>A</td>
      <td>N</td>
      <td>...</td>
      <td>0.686964</td>
      <td>0.420667</td>
      <td>0.648182</td>
      <td>0.684501</td>
      <td>0.956692</td>
      <td>1.000773</td>
      <td>0.776742</td>
      <td>0.625849</td>
      <td>0.250823</td>
      <td>7.972260</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>

## Preprocesamiento


{{< highlight "python" "linenos=false">}}
# Separamos el target de las features
y = train['target']
features = train.drop(['target'], axis=1)

# Previsualización de las features
features.head()
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
      <th>cat0</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>C</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>E</td>
      <td>C</td>
      <td>N</td>
      <td>...</td>
      <td>0.610706</td>
      <td>0.400361</td>
      <td>0.160266</td>
      <td>0.310921</td>
      <td>0.389470</td>
      <td>0.267559</td>
      <td>0.237281</td>
      <td>0.377873</td>
      <td>0.322401</td>
      <td>0.869850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>F</td>
      <td>A</td>
      <td>O</td>
      <td>...</td>
      <td>0.276853</td>
      <td>0.533087</td>
      <td>0.558922</td>
      <td>0.516294</td>
      <td>0.594928</td>
      <td>0.341439</td>
      <td>0.906013</td>
      <td>0.921701</td>
      <td>0.261975</td>
      <td>0.465083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>D</td>
      <td>A</td>
      <td>F</td>
      <td>...</td>
      <td>0.285074</td>
      <td>0.650609</td>
      <td>0.375348</td>
      <td>0.902567</td>
      <td>0.555205</td>
      <td>0.843531</td>
      <td>0.748809</td>
      <td>0.620126</td>
      <td>0.541474</td>
      <td>0.763846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>E</td>
      <td>C</td>
      <td>K</td>
      <td>...</td>
      <td>0.284667</td>
      <td>0.668980</td>
      <td>0.239061</td>
      <td>0.732948</td>
      <td>0.679618</td>
      <td>0.574844</td>
      <td>0.346010</td>
      <td>0.714610</td>
      <td>0.540150</td>
      <td>0.280682</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>A</td>
      <td>E</td>
      <td>A</td>
      <td>N</td>
      <td>...</td>
      <td>0.287595</td>
      <td>0.686964</td>
      <td>0.420667</td>
      <td>0.648182</td>
      <td>0.684501</td>
      <td>0.956692</td>
      <td>1.000773</td>
      <td>0.776742</td>
      <td>0.625849</td>
      <td>0.250823</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

Seleccionamos y transformamos las variables categóricas a valores numéricos, antes de entrenar y evaluar nuestro modelo. Para ello usamos **Ordinal Encoding**.


{{< highlight "python" "linenos=false">}}
# Lista de columnas categóricas
object_cols = [col for col in features.columns if 'cat' in col]

# Aplicamos ordinal-encode a las columnas categóricas
X = features.copy()
X_test = test.copy()
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(features[object_cols])
X_test[object_cols] = ordinal_encoder.transform(test[object_cols])

# Previsualización de las features con ordinal-encoded
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
      <th>cat0</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.610706</td>
      <td>0.400361</td>
      <td>0.160266</td>
      <td>0.310921</td>
      <td>0.389470</td>
      <td>0.267559</td>
      <td>0.237281</td>
      <td>0.377873</td>
      <td>0.322401</td>
      <td>0.869850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>...</td>
      <td>0.276853</td>
      <td>0.533087</td>
      <td>0.558922</td>
      <td>0.516294</td>
      <td>0.594928</td>
      <td>0.341439</td>
      <td>0.906013</td>
      <td>0.921701</td>
      <td>0.261975</td>
      <td>0.465083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.285074</td>
      <td>0.650609</td>
      <td>0.375348</td>
      <td>0.902567</td>
      <td>0.555205</td>
      <td>0.843531</td>
      <td>0.748809</td>
      <td>0.620126</td>
      <td>0.541474</td>
      <td>0.763846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>0.284667</td>
      <td>0.668980</td>
      <td>0.239061</td>
      <td>0.732948</td>
      <td>0.679618</td>
      <td>0.574844</td>
      <td>0.346010</td>
      <td>0.714610</td>
      <td>0.540150</td>
      <td>0.280682</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.287595</td>
      <td>0.686964</td>
      <td>0.420667</td>
      <td>0.648182</td>
      <td>0.684501</td>
      <td>0.956692</td>
      <td>1.000773</td>
      <td>0.776742</td>
      <td>0.625849</td>
      <td>0.250823</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



Extraemos un conjunto de validación a partir de los datos de entrenamiento:


{{< highlight "python" "linenos=false">}}
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
{{< /highlight >}}

## Intento 1 - Entrenamiento de un modelo Random Forest


{{< highlight "python" "linenos=false">}}
# Definimos el modelo
model_rf = RandomForestRegressor(random_state=1, n_jobs=-1)

# Entrenamiento del modelo (puede tarder unos minutos en terminar)
model_rf.fit(X_train, y_train)
preds_valid = model_rf.predict(X_valid)
rmse_rf = mean_squared_error(y_valid, preds_valid, squared=False)
print(rmse_rf)
{{< /highlight >}}

    0.7375392165180452
    

Bien, ya tenemos nuestro primer resultado. Estoy ansioso por realizar el proceso completo y comprobar mi posición en la clasificación. Así que sin más demora me lanzo a realizar mi primer *submit*.


{{< highlight "python" "linenos=false">}}
# Usamos el modelo entrenado para realizar predicciones
predictions = model_rf.predict(X_test)

# Guardamos la predicciones en un archivo CSV, según las instrucciones de la competición
output = pd.DataFrame({'Id': X_test.index,
                       'target': predictions})
output.to_csv('output/submission.csv', index=False)
{{< /highlight >}}

Cuando enviamos dicho archivo nos indican cuál es la puntuación obtenida (*public score*). Kaggle calcula dicha puntuación usando solo una parte de los datos de prueba. La puntuación final (*private score*) se calculará usando el conjunto completo de prueba. La puntuación privada no será visible para nosotros ni para ninguno de los competidores y solo la conoceremos al final de la competición.

La puntuación pública obtenida es de **0.73845**. Esta puntuación es resultado de entrenar un modelo Random Forest con los hiperparámetros por defecto, por lo tanto, nuestra posición en la clasificación se ubica en la parte baja de la tabla, igualada a la de otros miles de competidores (en total participamos 7.500 equipos). Por tanto, todavía tenemos mucho margen para seguir mejorando.

## Intento 2 - Entrenamiento de un modelo XGBoost

Vamos a entrenar unos de los modelos "estrella" en muchas de las competiciones de Kaggle: XGBoost.


{{< highlight "python" "linenos=false">}}
# Definimos el modelo
model_xgb = XGBRegressor(random_state=1, n_jobs=-1)

# Entrenamiento del modelo
model_xgb.fit(X_train, y_train)
preds_valid = model_xgb.predict(X_valid)
rmse_xgb = mean_squared_error(y_valid, preds_valid, squared=False)
print(rmse_xgb)
{{< /highlight >}}

    0.7268784689736293
    

Bueno, hemos mejorado ligeramente respecto al uso de Random Forest. Así que como hicimos anteriormente, generamos nuestra predicciones, exportamos nuestro archivo de envío y lo subimos a Kaggle para ver nuestra puntuación.


{{< highlight "python" "linenos=false">}}
# Usamos el modelo entrenado para realizar predicciones
predictions = model_xgb.predict(X_test)

# Guardamos la predicciones en un archivo CSV, según las instrucciones de la competición
output = pd.DataFrame({'Id': X_test.index,
                       'target': predictions})
output.to_csv('output/submission.csv', index=False)
{{< /highlight >}}

La puntuación pública obtenida es de **0.72613**. Son solo unas décimas respecto al envío previo, pero suficientes para escalar a la zona media de la tabla. Seguro que podemos hacerlo mejor... por ejemplo, afinar algunos hiperparámetros. Vamos a ello.

## Intento 3 - Entrenamiento de un modelo XGBoost - Refinamiento usando Grid Search

Para este refinamiento, vamos a usar GridSearch para encontrar cuál es la mejor combinación de algunos hiperparámetros.


{{< highlight "python" "linenos=false">}}
# Definimos el modelo
model_xgb = XGBRegressor(random_state=1)
clf = GridSearchCV(model_xgb,
                   {'max_depth': [2, 4, 6],
                    'n_estimators': [50, 100, 200, 500]}, 
                   scoring='neg_root_mean_squared_error',
                   verbose=1, n_jobs=1)
clf.fit(X_train, y_train)
print(-clf.best_score_)
print(clf.best_params_)
{{< /highlight >}}

    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    0.7201387194775795
    {'max_depth': 2, 'n_estimators': 500}
    


{{< highlight "python" "linenos=false">}}
# Entrenamiento del modelo con los mejores parámetros
preds_valid = clf.predict(X_valid)
rmse_xgb = mean_squared_error(y_valid, preds_valid, squared=False)
print(rmse_xgb)
{{< /highlight >}}

    0.7221846356377921
    

Bien, otra ligera mejora. Así que como hicimos anteriormente, generamos nuestra predicciones, exportamos nuestro archivo de envío y lo subimos a Kaggle para ver nuestra puntuación.


{{< highlight "python" "linenos=false">}}
# Usamos el modelo entrenado para realizar predicciones
predictions = clf.predict(X_test)

# Guardamos la predicciones en un archivo CSV, según las instrucciones de la competición
output = pd.DataFrame({'Id': X_test.index,
                       'target': predictions})
output.to_csv('output/submission.csv', index=False)
{{< /highlight >}}

La puntuación pública obtenida es de **0.72181**. Igualmente, son solo unas décimas respecto al envío previo, pero suficientes para seguir escalando posiciones. Hemos superado la zona media de la tabla Sigamos afinando algunos hiperparámetros.

## Intento 4 (y último) - Entrenamiento de un modelo XGBoost - Refinamiento usando Grid Search

Seguimos ajustando hiperparámetros. Dado que finalmente el mejor valor para `n_estimators` era `500`, lo que representaba el limite superior de la lista proporcionada, vamos a seguir probando más alla de este límite.


{{< highlight "python" "linenos=false">}}
# Definimos el modelo
model_xgb = XGBRegressor(random_state=1)
clf = GridSearchCV(model_xgb,
                   {'max_depth': [2],
                    'n_estimators': [500, 1000, 2000, 3000]}, 
                   scoring='neg_root_mean_squared_error',
                   verbose=2, n_jobs=1)
clf.fit(X_train, y_train)
print(-clf.best_score_)
print(clf.best_params_)
{{< /highlight >}}

    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    [CV] END ......................max_depth=2, n_estimators=500; total time=  16.7s
    [CV] END ......................max_depth=2, n_estimators=500; total time=  15.8s
    [CV] END ......................max_depth=2, n_estimators=500; total time=  15.6s
    [CV] END ......................max_depth=2, n_estimators=500; total time=  15.7s
    [CV] END ......................max_depth=2, n_estimators=500; total time=  16.0s
    [CV] END .....................max_depth=2, n_estimators=1000; total time=  32.1s
    [CV] END .....................max_depth=2, n_estimators=1000; total time=  31.3s
    [CV] END .....................max_depth=2, n_estimators=1000; total time=  31.4s
    [CV] END .....................max_depth=2, n_estimators=1000; total time=  31.2s
    [CV] END .....................max_depth=2, n_estimators=1000; total time=  31.1s
    [CV] END .....................max_depth=2, n_estimators=2000; total time= 1.1min
    [CV] END .....................max_depth=2, n_estimators=2000; total time= 1.0min
    [CV] END .....................max_depth=2, n_estimators=2000; total time= 1.1min
    [CV] END .....................max_depth=2, n_estimators=2000; total time= 1.1min
    [CV] END .....................max_depth=2, n_estimators=2000; total time= 1.1min
    [CV] END .....................max_depth=2, n_estimators=3000; total time= 1.6min
    [CV] END .....................max_depth=2, n_estimators=3000; total time= 1.6min
    [CV] END .....................max_depth=2, n_estimators=3000; total time= 1.6min
    [CV] END .....................max_depth=2, n_estimators=3000; total time= 1.6min
    [CV] END .....................max_depth=2, n_estimators=3000; total time= 1.7min
    0.7194840350716621
    {'max_depth': 2, 'n_estimators': 1000}


{{< highlight "python" "linenos=false">}}
# Entrenamiento del modelo con los mejores parámetros
preds_valid = clf.predict(X_valid)
rmse_xgb = mean_squared_error(y_valid, preds_valid, squared=False)
print(rmse_xgb)
{{< /highlight >}}

    0.7211765635584879
    


{{< highlight "python" "linenos=false">}}
# Usamos el modelo entrenado para realizar predicciones
predictions = clf.predict(X_test)

# Guardamos la predicciones en un archivo CSV, según las instrucciones de la competición
output = pd.DataFrame({'Id': X_test.index,
                       'target': predictions})
output.to_csv('output/submission.csv', index=False)
{{< /highlight >}}

La puntuación pública obtenida es de **0.72028**. Mejoramos ligeramente y subimos posiciones en la clasificación. Finalmente no puedo dedicarle más tiempo (los ciclos de entrenamiento llevan su tiempo) y Kaggle comunica la finalización del evento. Publica las puntuaciones privadas, calculadas sobre la totalidad de los datos de prueba: el score final obtenido es **0.71874**. La posición final en la clasificación es 2780 sobre un total de 7572 concursantes. Los diez primeros clasificados se encuentran en una horquilla de 0.71533 a 0.71547. 

En fin, no está mal. Sin tener más información sobre las features y su significado, podríamos seguir empleando fuerza bruta, potencia de cálculo y tiempo para seguir afinando hiperparámetros con el objetivo de seguir disminuyendo algunas milésimas a la métrica.
