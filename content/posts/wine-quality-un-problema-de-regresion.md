---
title: "Calidad del vino - Un problema de regresión"
date: 2021-05-05T16:37:06+02:00
draft: False
---

En este post repasaremos las principales fases que componen un proyecto de Machine Learning.

Existen ocho pasos principales:

1. Encuadrar el problema y disponer de la visión global.

2. Obtener los datos.

3. Explorar los datos para obtener ideas.

4. Preparar los datos para exponer lo mejor posible los patrones de datos subyacentes a los algoritmos de Machine Learning.

5. Explorar muchos modelos diferentes y preseleccionar los mejores.

6. Afinar nuestros modelos y combinarlos en una gran solución.

7. Presentar nuestra solución.

8. Implantar, monitorizar y mantener nuestro sistema.

Disponemos un conjunto de datos que contiene diversas características de variantes de tinto y blanco del vino portugués "Vinho Verde". Disponemos de variables químicas, como son la cantidad de alcohol, ácido cítrico, acidez, densidad, pH, etc; así como de una variable sensorial y subjetiva como es la puntuación con la que un grupo de expertos calificaron la calidad del vino: entre 0 (muy malo) y 10 (muy excelente).

El objetivo es desarrollar un modelo que pueda predecir la puntuación de calidad dados dichos indicadores bioquímicos.

Lo primero que nos viene a la mente son una serie de preguntas básicas:

+ ¿Cómo se enmarcaría este problema (supervisado, no supervisado, etc.)?

+ ¿Cuál es la variable objetivo? ¿Cuáles son los predictores?

+ ¿Cómo vamos a medir el rendimiento de nuestro modelo?

![wine glasses](/images/wine-glasses-1246240_1280.jpg)

El codigo python utilizado en este artículo está disponible en mi [repositorio github](https://github.com/SgtSteiner/DataScience/blob/master/Wine%20Quality/red-wine-quality-regression_v3.ipynb)

En primer lugar importamos todas las librerías necesarias:

{{< highlight "python" "linenos=false">}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn import metrics

%matplotlib inline
{{< /highlight >}}

## Get the Data


{{< highlight "python" "linenos=false">}}
red = pd.read_csv("data/wine-quality/winequality-red.csv")
{{< /highlight >}}

### Check the size and type of data


{{< highlight "python" "linenos=false">}}
red.shape
{{< /highlight >}}

    (1599, 12)


{{< highlight "python" "linenos=false">}}
red.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


{{< highlight "python" "linenos=false">}}
red.info()
{{< /highlight >}}

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB
    


{{< highlight "python" "linenos=false">}}
pd.DataFrame({"Type": red.dtypes,
              "Unique": red.nunique(),
              "Null": red.isnull().sum(),
              "Null percent": red.isnull().sum() / len(red),
              "Mean": red.mean(),
              "Std": red.std()})
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
      <th>Type</th>
      <th>Unique</th>
      <th>Null</th>
      <th>Null percent</th>
      <th>Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>float64</td>
      <td>96</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.319637</td>
      <td>1.741096</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>float64</td>
      <td>143</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.527821</td>
      <td>0.179060</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>float64</td>
      <td>80</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.270976</td>
      <td>0.194801</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>float64</td>
      <td>91</td>
      <td>0</td>
      <td>0.0</td>
      <td>2.538806</td>
      <td>1.409928</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>float64</td>
      <td>153</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.087467</td>
      <td>0.047065</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>float64</td>
      <td>60</td>
      <td>0</td>
      <td>0.0</td>
      <td>15.874922</td>
      <td>10.460157</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>float64</td>
      <td>144</td>
      <td>0</td>
      <td>0.0</td>
      <td>46.467792</td>
      <td>32.895324</td>
    </tr>
    <tr>
      <th>density</th>
      <td>float64</td>
      <td>436</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.996747</td>
      <td>0.001887</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>float64</td>
      <td>89</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.311113</td>
      <td>0.154386</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>float64</td>
      <td>96</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.658149</td>
      <td>0.169507</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>float64</td>
      <td>65</td>
      <td>0</td>
      <td>0.0</td>
      <td>10.422983</td>
      <td>1.065668</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>int64</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>5.636023</td>
      <td>0.807569</td>
    </tr>
  </tbody>
</table>
</div>


Mmmmm, there are no nulls, what a data set!


{{< highlight "python" "linenos=false">}}
red.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>1599.0</td>
      <td>8.319637</td>
      <td>1.741096</td>
      <td>4.60000</td>
      <td>7.1000</td>
      <td>7.90000</td>
      <td>9.200000</td>
      <td>15.90000</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>1599.0</td>
      <td>0.527821</td>
      <td>0.179060</td>
      <td>0.12000</td>
      <td>0.3900</td>
      <td>0.52000</td>
      <td>0.640000</td>
      <td>1.58000</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>1599.0</td>
      <td>0.270976</td>
      <td>0.194801</td>
      <td>0.00000</td>
      <td>0.0900</td>
      <td>0.26000</td>
      <td>0.420000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>1599.0</td>
      <td>2.538806</td>
      <td>1.409928</td>
      <td>0.90000</td>
      <td>1.9000</td>
      <td>2.20000</td>
      <td>2.600000</td>
      <td>15.50000</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>1599.0</td>
      <td>0.087467</td>
      <td>0.047065</td>
      <td>0.01200</td>
      <td>0.0700</td>
      <td>0.07900</td>
      <td>0.090000</td>
      <td>0.61100</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>1599.0</td>
      <td>15.874922</td>
      <td>10.460157</td>
      <td>1.00000</td>
      <td>7.0000</td>
      <td>14.00000</td>
      <td>21.000000</td>
      <td>72.00000</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>1599.0</td>
      <td>46.467792</td>
      <td>32.895324</td>
      <td>6.00000</td>
      <td>22.0000</td>
      <td>38.00000</td>
      <td>62.000000</td>
      <td>289.00000</td>
    </tr>
    <tr>
      <th>density</th>
      <td>1599.0</td>
      <td>0.996747</td>
      <td>0.001887</td>
      <td>0.99007</td>
      <td>0.9956</td>
      <td>0.99675</td>
      <td>0.997835</td>
      <td>1.00369</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>1599.0</td>
      <td>3.311113</td>
      <td>0.154386</td>
      <td>2.74000</td>
      <td>3.2100</td>
      <td>3.31000</td>
      <td>3.400000</td>
      <td>4.01000</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>1599.0</td>
      <td>0.658149</td>
      <td>0.169507</td>
      <td>0.33000</td>
      <td>0.5500</td>
      <td>0.62000</td>
      <td>0.730000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>1599.0</td>
      <td>10.422983</td>
      <td>1.065668</td>
      <td>8.40000</td>
      <td>9.5000</td>
      <td>10.20000</td>
      <td>11.100000</td>
      <td>14.90000</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>1599.0</td>
      <td>5.636023</td>
      <td>0.807569</td>
      <td>3.00000</td>
      <td>5.0000</td>
      <td>6.00000</td>
      <td>6.000000</td>
      <td>8.00000</td>
    </tr>
  </tbody>
</table>
</div>



## Explore the Data

How are the features distributed?


{{< highlight "python" "linenos=false">}}
red.hist(bins=50, figsize=(15,12));
{{< /highlight >}}


    
![png](/images/output_19_0.png)
    


Let's check how our target variable, the quality score, is distributed:


{{< highlight "python" "linenos=false">}}
print(f"Percentage of quality scores")
red["quality"].value_counts(normalize=True) * 100
{{< /highlight >}}

    Percentage of quality scores
    
    5    42.589118
    6    39.899937
    7    12.445278
    4     3.314572
    8     1.125704
    3     0.625391
    Name: quality, dtype: float64



It is significantly unbalanced. Most instances (82%) have scores of 6 or 5.

We are going to check the correlations between the attributes of the dataset:


{{< highlight "python" "linenos=false">}}
corr_matrix = red.corr()
corr_matrix
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>1.000000</td>
      <td>-0.256131</td>
      <td>0.671703</td>
      <td>0.114777</td>
      <td>0.093705</td>
      <td>-0.153794</td>
      <td>-0.113181</td>
      <td>0.668047</td>
      <td>-0.682978</td>
      <td>0.183006</td>
      <td>-0.061668</td>
      <td>0.124052</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>-0.256131</td>
      <td>1.000000</td>
      <td>-0.552496</td>
      <td>0.001918</td>
      <td>0.061298</td>
      <td>-0.010504</td>
      <td>0.076470</td>
      <td>0.022026</td>
      <td>0.234937</td>
      <td>-0.260987</td>
      <td>-0.202288</td>
      <td>-0.390558</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>0.671703</td>
      <td>-0.552496</td>
      <td>1.000000</td>
      <td>0.143577</td>
      <td>0.203823</td>
      <td>-0.060978</td>
      <td>0.035533</td>
      <td>0.364947</td>
      <td>-0.541904</td>
      <td>0.312770</td>
      <td>0.109903</td>
      <td>0.226373</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>0.114777</td>
      <td>0.001918</td>
      <td>0.143577</td>
      <td>1.000000</td>
      <td>0.055610</td>
      <td>0.187049</td>
      <td>0.203028</td>
      <td>0.355283</td>
      <td>-0.085652</td>
      <td>0.005527</td>
      <td>0.042075</td>
      <td>0.013732</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.093705</td>
      <td>0.061298</td>
      <td>0.203823</td>
      <td>0.055610</td>
      <td>1.000000</td>
      <td>0.005562</td>
      <td>0.047400</td>
      <td>0.200632</td>
      <td>-0.265026</td>
      <td>0.371260</td>
      <td>-0.221141</td>
      <td>-0.128907</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>-0.153794</td>
      <td>-0.010504</td>
      <td>-0.060978</td>
      <td>0.187049</td>
      <td>0.005562</td>
      <td>1.000000</td>
      <td>0.667666</td>
      <td>-0.021946</td>
      <td>0.070377</td>
      <td>0.051658</td>
      <td>-0.069408</td>
      <td>-0.050656</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>-0.113181</td>
      <td>0.076470</td>
      <td>0.035533</td>
      <td>0.203028</td>
      <td>0.047400</td>
      <td>0.667666</td>
      <td>1.000000</td>
      <td>0.071269</td>
      <td>-0.066495</td>
      <td>0.042947</td>
      <td>-0.205654</td>
      <td>-0.185100</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.668047</td>
      <td>0.022026</td>
      <td>0.364947</td>
      <td>0.355283</td>
      <td>0.200632</td>
      <td>-0.021946</td>
      <td>0.071269</td>
      <td>1.000000</td>
      <td>-0.341699</td>
      <td>0.148506</td>
      <td>-0.496180</td>
      <td>-0.174919</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.682978</td>
      <td>0.234937</td>
      <td>-0.541904</td>
      <td>-0.085652</td>
      <td>-0.265026</td>
      <td>0.070377</td>
      <td>-0.066495</td>
      <td>-0.341699</td>
      <td>1.000000</td>
      <td>-0.196648</td>
      <td>0.205633</td>
      <td>-0.057731</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.183006</td>
      <td>-0.260987</td>
      <td>0.312770</td>
      <td>0.005527</td>
      <td>0.371260</td>
      <td>0.051658</td>
      <td>0.042947</td>
      <td>0.148506</td>
      <td>-0.196648</td>
      <td>1.000000</td>
      <td>0.093595</td>
      <td>0.251397</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>-0.061668</td>
      <td>-0.202288</td>
      <td>0.109903</td>
      <td>0.042075</td>
      <td>-0.221141</td>
      <td>-0.069408</td>
      <td>-0.205654</td>
      <td>-0.496180</td>
      <td>0.205633</td>
      <td>0.093595</td>
      <td>1.000000</td>
      <td>0.476166</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>0.124052</td>
      <td>-0.390558</td>
      <td>0.226373</td>
      <td>0.013732</td>
      <td>-0.128907</td>
      <td>-0.050656</td>
      <td>-0.185100</td>
      <td>-0.174919</td>
      <td>-0.057731</td>
      <td>0.251397</td>
      <td>0.476166</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




{{< highlight "python" "linenos=false">}}
plt.figure(figsize=(15,10))
sns.heatmap(red.corr(), annot=True, cmap='coolwarm')
plt.show()
{{< /highlight >}}


    
![png](/images/output_25_0.png)
    


We show only the correlations of the target variable with the rest of the attributes:


{{< highlight "python" "linenos=false">}}
corr_matrix["quality"].drop("quality").sort_values(ascending=False)
{{< /highlight >}}

    alcohol                 0.476166
    sulphates               0.251397
    citric acid             0.226373
    fixed acidity           0.124052
    residual sugar          0.013732
    free sulfur dioxide    -0.050656
    pH                     -0.057731
    chlorides              -0.128907
    density                -0.174919
    total sulfur dioxide   -0.185100
    volatile acidity       -0.390558
    Name: quality, dtype: float64




{{< highlight "python" "linenos=false">}}
plt.figure(figsize=(8,5))
corr_matrix["quality"].drop("quality").sort_values(ascending=False).plot(kind='bar')
plt.title("Attribute correlations with quality")
plt.show()
{{< /highlight >}}


    
![png](/images/output_28_0.png)
    


## Prepare the Data

Create the predictor set and the set with the target variable:


{{< highlight "python" "linenos=false">}}
predict_columns = red.columns[:-1]
predict_columns
{{< /highlight >}}

    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
          dtype='object')


{{< highlight "python" "linenos=false">}}
X = red[predict_columns]
y = red["quality"]
{{< /highlight >}}

Create the training and test datasets:


{{< highlight "python" "linenos=false">}}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
X_train.shape, y_train.shape
{{< /highlight >}}

    ((1279, 11), (1279,))


{{< highlight "python" "linenos=false">}}
X_test.shape, y_test.shape
{{< /highlight >}}

    ((320, 11), (320,))



## Baseline


{{< highlight "python" "linenos=false">}}
def evaluate_model(estimator, X_train, y_train, cv=10, verbose=True):
    """Print and return cross validation of model
    """
    scoring = ["neg_mean_absolute_error", "neg_mean_squared_error", "r2"]
    scores = cross_validate(estimator, X_train, y_train, return_train_score=True, cv=cv, scoring=scoring)
    
    val_mae_mean, val_mae_std = -scores['test_neg_mean_absolute_error'].mean(), \
                                -scores['test_neg_mean_absolute_error'].std()
    
    train_mae_mean, train_mae_std = -scores['train_neg_mean_absolute_error'].mean(), \
                                    -scores['train_neg_mean_absolute_error'].std()
    
    val_mse_mean, val_mse_std = -scores['test_neg_mean_squared_error'].mean(), \
                                -scores['test_neg_mean_squared_error'].std()
    
    train_mse_mean, train_mse_std = -scores['train_neg_mean_squared_error'].mean(), \
                                    -scores['train_neg_mean_squared_error'].std()
    
    val_rmse_mean, val_rmse_std = np.sqrt(-scores['test_neg_mean_squared_error']).mean(), \
                                  np.sqrt(-scores['test_neg_mean_squared_error']).std()
    
    train_rmse_mean, train_rmse_std = np.sqrt(-scores['train_neg_mean_squared_error']).mean(), \
                                      np.sqrt(-scores['train_neg_mean_squared_error']).std()
    
    val_r2_mean, val_r2_std = scores['test_r2'].mean(), scores['test_r2'].std()
    
    train_r2_mean, train_r2_std = scores['train_r2'].mean(), scores['train_r2'].std()

    
    result = {
        "Val MAE": val_mae_mean,
        "Val MAE std": val_mae_std,
        "Train MAE": train_mae_mean,
        "Train MAE std": train_mae_std,
        "Val MSE": val_mse_mean,
        "Val MSE std": val_mse_std,
        "Train MSE": train_mse_mean,
        "Train MSE std": train_mse_std,
        "Val RMSE": val_rmse_mean,
        "Val RMSE std": val_rmse_std,
        "Train RMSE": train_rmse_mean,
        "Train RMSE std": train_rmse_std,
        "Val R2": val_r2_mean,
        "Val R2 std": val_r2_std,
        "Train R2": train_rmse_mean,
        "Train R2 std": train_r2_std,
    }
    
    if verbose:
        print(f"val_MAE_mean: {val_mae_mean} - (std: {val_mae_std})")
        print(f"train_MAE_mean: {train_mae_mean} - (std: {train_mae_std})")
        print(f"val_MSE_mean: {val_mse_mean} - (std: {val_mse_std})")
        print(f"train_MSE_mean: {train_mse_mean} - (std: {train_mse_std})")
        print(f"val_RMSE_mean: {val_rmse_mean} - (std: {val_rmse_std})")
        print(f"train_RMSE_mean: {train_rmse_mean} - (std: {train_rmse_std})")
        print(f"val_R2_mean: {val_r2_mean} - (std: {val_r2_std})")
        print(f"train_R2_mean: {train_r2_mean} - (std: {train_r2_std})")

    return result
{{< /highlight >}}

First, we are going to train a dummy regressor that we will use as a baseline with which to compare.


{{< highlight "python" "linenos=false">}}
rg_dummy = DummyRegressor(strategy="constant", constant=5) # Mean prediction
rg_dummy.fit(X_train, y_train)
{{< /highlight >}}

    DummyRegressor(constant=array(5), strategy='constant')


{{< highlight "python" "linenos=false">}}
rg_scores = evaluate_model(rg_dummy, X_train, y_train)
{{< /highlight >}}

    val_MAE_mean: 0.719365157480315 - (std: -0.06352462970037416)
    train_MAE_mean: 0.7193126146346173 - (std: -0.007057414168822716)
    val_MSE_mean: 1.0398868110236221 - (std: -0.12176257291946108)
    train_MSE_mean: 1.0398750482672072 - (std: -0.01354074583910719)
    val_RMSE_mean: 1.0180017820772593 - (std: 0.05965888627141756)
    train_RMSE_mean: 1.0197209977802941 - (std: 0.006643414270421584)
    val_R2_mean: -0.6192850555554466 - (std: 0.14799333040101653)
    train_R2_mean: -0.5986022943608599 - (std: 0.01598456942915052)
    

A classifier that always predicts the most frequent quality (in our case the quality score 5) obtains a RMSE = 1.039.


{{< highlight "python" "linenos=false">}}
rg_dummy = DummyRegressor(strategy="mean") # Mean prediction
rg_dummy.fit(X_train, y_train)
{{< /highlight >}}

    DummyRegressor()


{{< highlight "python" "linenos=false">}}
rg_scores = evaluate_model(rg_dummy, X_train, y_train)
{{< /highlight >}}

    val_MAE_mean: 0.6842639509806605 - (std: -0.039939453843720794)
    train_MAE_mean: 0.6836374055181736 - (std: -0.004461928774514038)
    val_MSE_mean: 0.6515564887161005 - (std: -0.08938937463665708)
    train_MSE_mean: 0.6505431870574859 - (std: -0.009928873673332832)
    val_RMSE_mean: 0.8052590895459458 - (std: 0.05580580095057208)
    train_RMSE_mean: 0.8065390950374436 - (std: 0.006154285796714715)
    val_R2_mean: -0.007632943779434287 - (std: 0.010684535533448955)
    train_R2_mean: 0.0 - (std: 0.0)
    

A regressor that always predicts the mean quality obtains a RMSE = 0.651. We are going to take the prediction of this dummy classifier as our baseline.

## Shortlist Promising Models

OK, we're going train several quick-and-dirty models from different categories using standard parameters. We selected some of the regression models: Linear Regression, Lasso, ElasticNet, Ridge, Extre Trees, and RandomForest.


{{< highlight "python" "linenos=false">}}
models = [LinearRegression(), Lasso(alpha=0.1), ElasticNet(),
          Ridge(), ExtraTreesRegressor(), RandomForestRegressor()]

model_names = ["Lineal Regression", "Lasso", "ElasticNet",
               "Ridge", "Extra Tree", "Random Forest"]
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
mae = []
mse = []
rmse = []
r2 = []

for model in range(len(models)):
    print(f"Paso {model+1} de {len(models)}")
    print(f"...running {model_names[model]}")
    
    rg_scores = evaluate_model(models[model], X_train, y_train)
    
    mae.append(rg_scores["Val MAE"])
    mse.append(rg_scores["Val MSE"])
    rmse.append(rg_scores["Val RMSE"])
    r2.append(rg_scores["Val R2"])
{{< /highlight >}}

    Paso 1 de 6
    ...running Lineal Regression
    val_MAE_mean: 0.5054157041773433 - (std: -0.046264972549372924)
    train_MAE_mean: 0.49951141240221786 - (std: -0.005396834677886112)
    val_MSE_mean: 0.4363366846653876 - (std: -0.0713599197838867)
    train_MSE_mean: 0.423559916011364 - (std: -0.007783364048942027)
    val_RMSE_mean: 0.6578988186927084 - (std: 0.059210041615646476)
    train_RMSE_mean: 0.6507877560250832 - (std: 0.005934022177307515)
    val_R2_mean: 0.32302131635332426 - (std: 0.0972958323285871)
    train_R2_mean: 0.34888336017832816 - (std: 0.008988207786517072)
    Paso 2 de 6
    ...running Lasso
    val_MAE_mean: 0.5542159398138832 - (std: -0.044044881537899525)
    train_MAE_mean: 0.551926769360105 - (std: -0.005222359881914205)
    val_MSE_mean: 0.5011613158962728 - (std: -0.07980261731926688)
    train_MSE_mean: 0.49648903729654775 - (std: -0.00886434349442919)
    val_RMSE_mean: 0.7054560563903938 - (std: 0.05910218607112876)
    train_RMSE_mean: 0.7045920060170998 - (std: 0.006256385006291075)
    val_R2_mean: 0.22550457016915199 - (std: 0.06858817248045986)
    train_R2_mean: 0.23679715721911138 - (std: 0.008061051196907644)
    Paso 3 de 6
    ...running ElasticNet
    val_MAE_mean: 0.6484828644185054 - (std: -0.03858618665902155)
    train_MAE_mean: 0.6472074434172257 - (std: -0.004861676284701619)
    val_MSE_mean: 0.6260699925252777 - (std: -0.08837053843631361)
    train_MSE_mean: 0.6236958050351286 - (std: -0.009753039023728842)
    val_RMSE_mean: 0.7891968495348196 - (std: 0.056906284447264595)
    train_RMSE_mean: 0.7897200517246066 - (std: 0.0061680579774354895)
    val_R2_mean: 0.032300440343033296 - (std: 0.027013749786509673)
    train_R2_mean: 0.041268269123349036 - (std: 0.0034334107542665303)
    Paso 4 de 6
    ...running Ridge
    val_MAE_mean: 0.5052017417711606 - (std: -0.04639189777979148)
    train_MAE_mean: 0.5000120146851917 - (std: -0.00538293390792397)
    val_MSE_mean: 0.4353611411950837 - (std: -0.07150445371257734)
    train_MSE_mean: 0.4243933932521361 - (std: -0.007774091981744382)
    val_RMSE_mean: 0.6571341500690723 - (std: 0.05946301378236467)
    train_RMSE_mean: 0.6514279204128516 - (std: 0.0059209592739344254)
    val_R2_mean: 0.32476443307512515 - (std: 0.09605257129964452)
    train_R2_mean: 0.3476024511130947 - (std: 0.0089301257345918)
    Paso 5 de 6
    ...running Extra Tree
    val_MAE_mean: 0.3767233021653543 - (std: -0.048411131876621855)
    train_MAE_mean: -0.0 - (std: -0.0)
    val_MSE_mean: 0.33849758981299216 - (std: -0.07037684927470149)
    train_MSE_mean: -0.0 - (std: -0.0)
    val_RMSE_mean: 0.5784725891678845 - (std: 0.062185636560190514)
    train_RMSE_mean: 0.0 - (std: 0.0)
    val_R2_mean: 0.4753582472917177 - (std: 0.09435328966382882)
    train_R2_mean: 1.0 - (std: 0.0)
    Paso 6 de 6
    ...running Random Forest
    val_MAE_mean: 0.421939406988189 - (std: -0.03848180259232641)
    train_MAE_mean: 0.15720688154624 - (std: -0.0024955091475250693)
    val_MSE_mean: 0.3536394728100393 - (std: -0.06315688035394738)
    train_MSE_mean: 0.049982221460505356 - (std: -0.0012897300801719821)
    val_RMSE_mean: 0.5921558699969636 - (std: 0.05468910712544724)
    train_RMSE_mean: 0.22354850027354933 - (std: 0.0028791467403151403)
    val_R2_mean: 0.450229047801475 - (std: 0.08970981370698214)
    train_R2_mean: 0.9231573754360927 - (std: 0.0020715859571618753)
    

Let's see the performance of each of them:


{{< highlight "python" "linenos=false">}}
df_result = pd.DataFrame({"Model": model_names,
                          "MAE": mae,
                          "MSE": mse,
                          "RMSE": rmse,
                          "R2": r2})
df_result
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
      <th>Model</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>RMSE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lineal Regression</td>
      <td>0.505416</td>
      <td>0.436337</td>
      <td>0.657899</td>
      <td>0.323021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lasso</td>
      <td>0.554216</td>
      <td>0.501161</td>
      <td>0.705456</td>
      <td>0.225505</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ElasticNet</td>
      <td>0.648483</td>
      <td>0.626070</td>
      <td>0.789197</td>
      <td>0.032300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ridge</td>
      <td>0.505202</td>
      <td>0.435361</td>
      <td>0.657134</td>
      <td>0.324764</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extra Tree</td>
      <td>0.376723</td>
      <td>0.338498</td>
      <td>0.578473</td>
      <td>0.475358</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.421939</td>
      <td>0.353639</td>
      <td>0.592156</td>
      <td>0.450229</td>
    </tr>
  </tbody>
</table>
</div>


{{< highlight "python" "linenos=false">}}
df_result.sort_values(by="RMSE", ascending=False).plot.barh("Model", "RMSE");
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
df_result.sort_values(by="R2").plot.barh("Model", "R2");
{{< /highlight >}}

The model that gives the best results is **extra trees**. RMSE = 0.577591 and R2 = 0.477845. Let's fine tune it.

## Fine-Tune


{{< highlight "python" "linenos=false">}}
param_grid = [
    {'n_estimators': range(10, 300, 10), 'max_features': [2, 3, 4, 5, 8, "auto"], 'bootstrap': [True, False]}
]


xtree_reg = ExtraTreesRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(xtree_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)

grid_search.fit(X_train, y_train)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
grid_search.best_params_
{{< /highlight >}}

It's the moment of truth! Let's see the performance on the test set:


{{< highlight "python" "linenos=false">}}
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
print(f"R2: {final_model.score(X_test, y_test)}")
{{< /highlight>}}

Well, a little better!


{{< highlight "python" "linenos=false">}}
plt.figure(figsize=(10,8))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.show()
{{< /highlight >}}

Let's see which features are most relevant:


{{< highlight "python" "linenos=false">}}
feature_importances = final_model.feature_importances_
feature_importances
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
sorted(zip(feature_importances, X_test.columns), reverse=True)
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
feature_imp = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)
feature_imp.plot(kind='bar')
plt.title('Feature Importances')
{{< /highlight >}}

Let's see how the errors are distributed:


{{< highlight "python" "linenos=false">}}
df_resul = pd.DataFrame({"Pred": y_pred,
              "Real": y_test,
              "error": y_pred - y_test,
              "error_abs": abs(y_pred - y_test)})
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
df_resul["error"].plot.hist(bins=40, density=True)
plt.title("Error distribution")
plt.xlabel("Error");
{{< /highlight >}}

More generally, What's the MAE that occurs in each quality score?


{{< highlight "python" "linenos=false">}}
df_resul.groupby("Real")["error_abs"].mean()
{{< /highlight >}}


{{< highlight "python" "linenos=false">}}
df_resul.groupby("Real")["error_abs"].mean().plot.bar()
plt.title("MAE distribution")
plt.ylabel("MAE")
plt.xlabel("Quality");
{{< /highlight >}}

## Conclusions

After testing various models, the one that provided the best results is ExtraTrees. After fine tuning it, we get a significant improvement.

The basic line regression model offers an R2: 0.323021 and RMSE: 0.657899. The Extra Tree model offers an R2: 0.529512 and RMSE: 0.570954. However, the R2 score is still very low. According to the value obtained from R2, our model can barely explain 52% of the variance. That is, the percentage of relationship between the variables that can be explained by our model is 52.95%.

According to the MAE distribution graph, we can see that our model is not good for extreme scores. In fact, it is not capable of predicting any score of 3 or 8. As we saw in the distribution of the target variable, it is very unbalanced, there are hardly any observations for the extreme values, so the model does not have enough data training for all quality scores.

As a final consideration, we should try to approach modeling as a classification problem, to evaluate if it offers better results than a regression problem. We will see it in part 2 and 3 of this analysis.
