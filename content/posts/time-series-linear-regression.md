---
title: "Series temporales: (1) Regresión lineal"
date: 2022-05-11T16:36:09+02:00
tags: [series temporales, regresión lineal, time dummy, lag feature, time-step feature, ]
categories: [series temporales]
---

# Regresión lineal con series temporales

En este tipo de problemas, el objetivo básico de la predicción son las series temporales, que son un conjunto de observaciones registradas a lo largo del tiempo. En las aplicaciones de pronóstico, las observaciones se registran con una frecuencia regular, como puede ser diaria o mensualmente.


{{< highlight "python" "linenos=false">}}
import pandas as pd

df = pd.read_csv(
    "../data/book_sales.csv",
    index_col="Date",
    parse_dates=["Date"],
).drop("Paperback", axis=1)

df.head()
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
      <th>Hardcover</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-04-01</th>
      <td>139</td>
    </tr>
    <tr>
      <th>2000-04-02</th>
      <td>128</td>
    </tr>
    <tr>
      <th>2000-04-03</th>
      <td>172</td>
    </tr>
    <tr>
      <th>2000-04-04</th>
      <td>139</td>
    </tr>
    <tr>
      <th>2000-04-05</th>
      <td>191</td>
    </tr>
  </tbody>
</table>
</div>



Esta serie registra el número de ventas de libros en una tienda durante 30 días. Por simplicidad, tiene una única columna de observaciones, `Hardcover` con un índice de tiempo `Date`.

Usaremos el algoritmo de regresión lineal para construir modelos predictivos. Estos algoritmos aprenden cómo hacer una suma ponderada a partir de sus variables de entrada. Para dos variables tendríamos:

`objetivo = peso_1 * feature_1 + peso_2 + feature_2 + bias`

Durante el entrenamiento, el algoritmo de regresión aprende los valores para los parámetros `peso_1`, `peso_2` y `bias` que mejor se ajustan al `objetivo`. A este algoritmo se le suele llamar *mínimos cuadrados ordinarios* ya que elige valores que minimizan el error cuadrático entre el objetivo y las predicciones. Los pesos también se denominan *coeficientes de regresión* y al `bias` también se le llama *intercept* porque nos dice dónde cruza el eje y la grafica de esta función.

### Features de paso de tiempo

Existen dos tipo de features únicas y distintivas de las series temporales: las variables de paso de tiempo (*time-step*) y las variables de *lag*.

Las features de paso de tiempo son variables que se pueden derivar directamente del índice de tiempo. La feature de paso de tiempo más básica es la dummy (*time dummy*), que cuenta el número de pasos de tiempo en las series desde el principio al final.


{{< highlight "python" "linenos=false">}}
import numpy as np

df["Time"] = np.arange(len(df.index))

df.head()
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
      <th>Hardcover</th>
      <th>Time</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-04-01</th>
      <td>139</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2000-04-02</th>
      <td>128</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2000-04-03</th>
      <td>172</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2000-04-04</th>
      <td>139</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2000-04-05</th>
      <td>191</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



La regresión lineal con la time dummy produce el siguiente modelo:

`objetivo = peso * time + bias`


{{< highlight "python" "linenos=false">}}
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot("Time", "Hardcover", data=df, color="0.75")
ax = sns.regplot(x="Time", y="Hardcover", data=df, ci=None,
                 scatter_kws=dict(color="0.25"))
ax.set_title("Ventas de libros");
{{< /highlight >}}





    
![png](/images/output_9_1.png)
    


Las features de paso de tiempo nos permiten modelar la **dependencia del tiempo**. Una serie es dependiente del tiempo si sus valores se pueden predecir desde el momento en que ocurrieron. En las series de nuestro ejemplo, podemos predecir que las ventas al final de mes son generalmente más altas que las ventas al principio del mes.

### Features de *lag*

Para hacer una variable de *lag* deslizamos las observaciones de las series del objetivo para que parezcan haber ocurrido más tarde en el tiempo. Aquí hemos creado una variable lag de 1-paso, aunque también es posible desplazar varios pasos.


{{< highlight "python" "linenos=false">}}
df["Lag_1"] = df["Hardcover"].shift(1)
df = df.reindex(columns=["Hardcover", "Lag_1"])

df.head()
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
      <th>Hardcover</th>
      <th>Lag_1</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-04-01</th>
      <td>139</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-04-02</th>
      <td>128</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>2000-04-03</th>
      <td>172</td>
      <td>128.0</td>
    </tr>
    <tr>
      <th>2000-04-04</th>
      <td>139</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>2000-04-05</th>
      <td>191</td>
      <td>139.0</td>
    </tr>
  </tbody>
</table>
</div>



La regresión lineal con la variable lag produce el siguiente modelo:

`objetivo = peso * lag + bias`

Entonces, las variables de lag nos permiten dibujar gráficas donde cada observación en una serie se dibuja contra la observación anterior.


{{< highlight "python" "linenos=false">}}
fig, ax = plt.subplots()
ax = sns.regplot(x="Lag_1", y="Hardcover", data=df, ci=None,
                 scatter_kws=dict(color="0.25"))
ax.set_aspect("equal")
ax.set_title("Gráfico lag de Ventas");
{{< /highlight >}}




    
![png](/images/output_16_1.png)
    


Podemos ver en el gráfico de lag que las ventas de un día (`Hardcover`) están correlacionadas con las ventas del día anterior (`Lag_1`). Cuando vemos una relación como ésta sabemos que una variable de lag será útil.

De forma más genérica, las features de lag nos permiten modelar la **dependencia en serie o serial**. Una serie temporal tiene dependencia serial cuando una observación se puede predecir a partir de las observaciones previas. En nuestro ejemplo, podemos predecir que ventas altas en un día, generalmente significan ventas altas en el siguiente día.

La adaptación de los algoritmos de machine learning a los problemas de series temporales se trata en gran medida con la ingeniería de features del índice de tiempo y los lags. Aunque estamos usando regresión lineal, estas variables serán útiles independientemente del algoritmo que seleccionemos para nuestras predicciones.

## Ejemplo - Tráfico túnel

El tráfico de túnel es una serie temporal que describe el número de vehículos que viajan a través del Túnel de Baregg en Suiza cada día desde noviembre 2002 a noviembre 2005. En este ejemplo, practicaremos aplicando regresión lineal a variables de paso de tiempo y variables lag.


{{< highlight "python" "linenos=false">}}
tunnel = pd.read_csv(
    "../data/tunnel.csv",
    index_col="Day",
    parse_dates=["Day"])

tunnel.to_period()

tunnel.head()
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
      <th>NumVehicles</th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003-11-01</th>
      <td>103536</td>
    </tr>
    <tr>
      <th>2003-11-02</th>
      <td>92051</td>
    </tr>
    <tr>
      <th>2003-11-03</th>
      <td>100795</td>
    </tr>
    <tr>
      <th>2003-11-04</th>
      <td>102352</td>
    </tr>
    <tr>
      <th>2003-11-05</th>
      <td>106569</td>
    </tr>
  </tbody>
</table>
</div>



Por defecto, Pandas crea un `DatetimeIndex` cuyo tipo es `Timestamp`, equivalente a `np.datetime64`, representando una serie temporal como una secuencia de medidas tomadas en un determinado momento. Un `PeriodIndex`, por otro lado, representa una serie temporal como una secuencia de cuantiles acumulados en periodos de tiempo. Los periodos suelen ser más fáciles de trabajar con ellos.

### Variable de paso de tiempo

Siempre que a la serie temporal no le falten fechas, podemos crear una time dummy contando la longitud de las series.


{{< highlight "python" "linenos=false">}}
df = tunnel.copy()

df["Time"] = np.arange(len(tunnel.index))

df.head()
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
      <th>NumVehicles</th>
      <th>Time</th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003-11-01</th>
      <td>103536</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2003-11-02</th>
      <td>92051</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2003-11-03</th>
      <td>100795</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2003-11-04</th>
      <td>102352</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2003-11-05</th>
      <td>106569</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Vamos a entrenar un modelo de regresión lineal.


{{< highlight "python" "linenos=false">}}
from sklearn.linear_model import LinearRegression

X = df.loc[:, ["Time"]]
y = df.loc[:, "NumVehicles"]

model = LinearRegression()
model.fit(X, y)

# Almacena las predicciones como una serie temporal con el mismo
# índice de tiempo que los datos de entrenamiento
y_pred = pd.Series(model.predict(X), index=X.index)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
y_pred
{{< /highlight >}}




    Day
    2003-11-01     98176.206344
    2003-11-02     98198.703794
    2003-11-03     98221.201243
    2003-11-04     98243.698693
    2003-11-05     98266.196142
                      ...      
    2005-11-12    114869.313898
    2005-11-13    114891.811347
    2005-11-14    114914.308797
    2005-11-15    114936.806247
    2005-11-16    114959.303696
    Length: 747, dtype: float64



Veamos cuáles son los coeficientes e intercept obtenidos:


{{< highlight "python" "linenos=false">}}
model.coef_, model.intercept_
{{< /highlight >}}




    (array([22.49744953]), 98176.20634409295)



Por tanto, el modelo creado realmente es, aproximadamente: `Vehicles = 22.5 * Time + 98176`. Al dibujar los valores obtenidos a lo largo del tiempo se muestra cómo la regresión lineal ajustada a la time dummy crea la línea de tendencia para esta ecuación.


{{< highlight "python" "linenos=false">}}
# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot del Tráfico del Túnel');
{{< /highlight >}}


    
![png](/images/output_31_2.png)
    


### Variable lag

Pandas proporciona un método simple para "lagear" una serie, el método `shift`.


{{< highlight "python" "linenos=false">}}
df["Lag_1"] = df["NumVehicles"].shift(1)
df.head()
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
      <th>NumVehicles</th>
      <th>Time</th>
      <th>Lag_1</th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003-11-01</th>
      <td>103536</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2003-11-02</th>
      <td>92051</td>
      <td>1</td>
      <td>103536.0</td>
    </tr>
    <tr>
      <th>2003-11-03</th>
      <td>100795</td>
      <td>2</td>
      <td>92051.0</td>
    </tr>
    <tr>
      <th>2003-11-04</th>
      <td>102352</td>
      <td>3</td>
      <td>100795.0</td>
    </tr>
    <tr>
      <th>2003-11-05</th>
      <td>106569</td>
      <td>4</td>
      <td>102352.0</td>
    </tr>
  </tbody>
</table>
</div>



Cuando creamos variables lag, necesitamos decidir qué hacer con los valores faltantes que se generan. Una opción es rellenarlos, quizas con 0.0 o con el primer valor conocido. En lugar de esto vamos a eliminar los valores faltantes, asegurándonos también de eliminar los valores del objetivo en las fechas correspondientes.


{{< highlight "python" "linenos=false">}}
from sklearn.linear_model import LinearRegression

X = df.loc[:, ["Lag_1"]]
X.dropna(inplace=True)
y = df.loc[:, "NumVehicles"]
y, X = y.align(X, join="inner")

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
{{< /highlight >}}

El diagrama de lag nos muestra cómo de bien somos capaces de ajustar la relación entre el número de vehículos de un día y el número del día anterior.


{{< highlight "python" "linenos=false">}}
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Diagrama de lag del Tráfico del Túnel');
{{< /highlight >}}


    
![png](/images/output_38_1.png)
    


¿Qué significa esta predicción de la variable lag sobre cómo de bien puede predecir las series a lo largo del tiempo? El siguiente gráfico temporal nos muestra cómo nuestros pronósticos de ahora responden al comportamiento de las series del pasado reciente.


{{< highlight "python" "linenos=false">}}
ax = y.plot(**plot_params)
ax = y_pred.plot()
{{< /highlight >}}


    
![png](/images/output_40_1.png)
    


Los mejores modelos de series temporales normalmente incluirán alguna combinación entre variables de paso de tiempo y variables lag.

# Ejercicio

Vamos a realizar un ejercicio para ampliar lo que acabamos de ver. Para ello cargaremos algunos datasets.


{{< highlight "python" "linenos=false">}}
book_sales = pd.read_csv(
    "../data/book_sales.csv",
    index_col="Date",
    parse_dates=["Date"],
).drop("Paperback", axis=1)
book_sales["Time"] = np.arange(len(book_sales.index))
book_sales["Lag_1"] = book_sales["Hardcover"].shift(1)
book_sales = book_sales.reindex(columns=["Hardcover", "Time", "Lag_1"])

book_sales.head()
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
      <th>Hardcover</th>
      <th>Time</th>
      <th>Lag_1</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-04-01</th>
      <td>139</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-04-02</th>
      <td>128</td>
      <td>1</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>2000-04-03</th>
      <td>172</td>
      <td>2</td>
      <td>128.0</td>
    </tr>
    <tr>
      <th>2000-04-04</th>
      <td>139</td>
      <td>3</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>2000-04-05</th>
      <td>191</td>
      <td>4</td>
      <td>139.0</td>
    </tr>
  </tbody>
</table>
</div>



Una de las ventajas que tiene la regresión lineal sobre algoritmos más complicados es que los modelos que genera son *interpretables*, es decir, es fácil interpretar la contribución que hace cada feature a las predicciones. En el modelo `objetivo = peso * feature + bias`, el `peso` nos dice cuánto cambia el `objetivo` de media por cada unidad de cambio de la `feature`.

Vamos a ver la regresión lineal de las ventas de libros:


{{< highlight "python" "linenos=false">}}
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Gráfico temporal de venta de libros');
{{< /highlight >}}


    
![png](/images/output_46_5.png)
    


## Interpretar la regresión lineal con *time dummy*

Digamos que la regresión lineal tiene una ecuación aproximada de: `Hardcover = 3.33 * Time + 150.5`. **Al cabo de 6 días, ¿cuánto se esperaría que cambiaran las ventas de libros?**

Si aplicamos la fórmula, entonces `3.33 * 6 + 150.5 = 19.98`. Luego se esperaría que las ventas sean de 19.98 libros. De acuerdo a este modelo, dado que la pendiente es 3.33, la venta de libros `Hardcover` cambiará de media 3.33 unidades por cada paso que cambie `Time`.


## Interpretar la regresión lineal con una variable lag

Interpretar los coeficientes de regresión puede ayudarnos a reconocer dependencias seriales en un gráfico temporal. Consideremos el modelo `objetivo = peso * lag_1 + error`, donde `error` es ruido aleatorio y `peso` es un número entre -1 y 1. En este caso, el `peso` nos dice cómo es de probable que el siguiente paso de tiempo tenga el mismo signo que el paso de tiempo anterior: un `peso` cercano a 1 significa que el `objetivo` probablemente tendrá el mismo signo que el paso previo, mientras que un `peso` cercano a -1 significa que el `objetivo` probablemente tendrá el signo opuesto.

Tenemos las siguientes dos series temporales:


{{< highlight "python" "linenos=false">}}
ar = pd.read_csv("../data/ar.csv")

ar.head()
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
      <th>ar1</th>
      <th>ar2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.541286</td>
      <td>-1.234475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.692950</td>
      <td>3.532498</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.730106</td>
      <td>-3.915508</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.783524</td>
      <td>2.820841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.796207</td>
      <td>-1.084120</td>
    </tr>
  </tbody>
</table>
</div>




{{< highlight "python" "linenos=false">}}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'])
ax1.set_title('Series 1')
ax2.plot(ar['ar2'])
ax2.set_title('Series 2');
{{< /highlight >}}


    
![png](/images/output_52_4.png)
    


Una de estas series tiene la ecuación: `objetivo = 0.95 * lag_1 + error` y la otra tiene la ecuación `objetivo = -0.95 * lag_1 + error`, diferenciándose únicamente por el signo de la variable lag. **¿Qué ecuación correspondería a cada serie?**

La Serie 1 estaría generada por la ecuación `objetivo = 0.95 * lag_1 + error` y la Serie 2 estaría generada por la ecuación `objetivo = -0.95 * lag_1 + error`. Como explicamos anteriormente, la serie con el peso 0.95 (signo positivo) tenderá a tener valores con signos que permanecen iguales. La serie con el peso -0.95 (signo negativo) tenderá a tener valores con signos que van y vienen.
`

## Entrenar una variable de paso de tiempo

Vamos a cargar el dataset de la competición de Pronóstico de series temporales de ventas de almacén. El dataset completo contiene casi 1800 series registrando las ventas de una amplia variedad de familias de productos desde 2013 a 2017. En principio solo trabajaremos con una única serie (`average_sales`) de las ventas medias por día.


{{< highlight "python" "linenos=false">}}
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    "../data/store_sales/train.csv",
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)

store_sales.head()
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
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>sales</th>
      <th>onpromotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>AUTOMOTIVE</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>BABY CARE</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>BEAUTY</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>BEVERAGES</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2013-01-01</td>
      <td>1</td>
      <td>BOOKS</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




{{< highlight "python" "linenos=false">}}
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)

store_sales.head()
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
      <th></th>
      <th></th>
      <th>id</th>
      <th>sales</th>
      <th>onpromotion</th>
    </tr>
    <tr>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2013-01-01</th>
      <th rowspan="5" valign="top">1</th>
      <th>AUTOMOTIVE</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BABY CARE</th>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BEAUTY</th>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BEVERAGES</th>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BOOKS</th>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




{{< highlight "python" "linenos=false">}}
average_sales = store_sales.groupby('date').mean()['sales']
average_sales.head()
{{< /highlight >}}




    date
    2013-01-01      1.409438
    2013-01-02    278.390808
    2013-01-03    202.840195
    2013-01-04    198.911148
    2013-01-05    267.873230
    Freq: D, Name: sales, dtype: float32



Vamos a crear un modelo de regresión lineal con una variable de paso de tiempo en la serie de promedio de ventas de producto. El objetivo es la columna `sales`.


{{< highlight "python" "linenos=false">}}
from sklearn.linear_model import LinearRegression

df = average_sales.to_frame()

# Crea time dummy
time = np.arange(len(df.index))

df['time'] = time

# Crea los datos de entrenamiento
X = df.loc[:, ["time"]]  # features
y = df.loc[:, "sales"]   # objetivo

# Entrena el modelo
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
{{< /highlight >}}

Vamos a dibujar la gráfica con el resultado:


{{< highlight "python" "linenos=false">}}
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Gráfica temporal de Ventas totales');
{{< /highlight >}}


    
![png](/images/output_63_3.png)
    


## Entrenar una variable lag

Vamos a crear un modelo de regresión lineal con una variable lag en la serie de promedio de ventas de producto. El objetivo es la columna `sales`.


{{< highlight "python" "linenos=false">}}
df = average_sales.to_frame()

# Crea la variable lag
lag_1 = df["sales"].shift(1)

df['lag_1'] = lag_1

X = df.loc[:, ['lag_1']].dropna()   # features
y = df.loc[:, 'sales']              # target
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=X.index)
{{< /highlight >}}

Vamos a dibujar la gráfica con el resultado:


{{< highlight "python" "linenos=false">}}
fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Diagrama de lag de Ventas promedio');
{{< /highlight >}}


    
![png](/images/output_68_1.png)
    

