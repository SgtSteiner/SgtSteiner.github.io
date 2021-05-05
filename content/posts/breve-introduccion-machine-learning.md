---
title: "Breve Introducción a Machine Learning"
date: 2020-07-01T17:51:42+02:00
draft: true
---

Lee Sedol tenía solo 12 años cuando se convirtió en uno de los jugadores profesionales de Go más jóvenes de la historia. Cuando el 9 de marzo de 2016 cruzó las puertas del Hotel Four Seasons de Seúl tenía 33 años y era 18 veces campeón del mundo. Le esperaban cinco intensas partidas contra un duro contrincante. Ante el asombro general perdió 4-1. Ese día pasaría a la historia como el día en que el campeón del mundo de Go perdió contra **AlphaGo, un programa informático perteneciente a la división DeepMind de Google**. Lee Sedol también pasaría a la historia como el único humano que ha ganado una partida a AlphaGo (aunque posteriormente reconocería que fue debido a un error en su programa).

![imagen go](/static/images/shutterstock_342026210.jpg)

Gran parte de la *magia negra* de AlphaGo proviene del uso de técnicas y sistemas de Machine Learning e Inteligencia Artificial. Los sistemas de **[Machine learning (ML)](https://en.wikipedia.org/wiki/Machine_learning)** o **aprendizaje automático**, están detrás de muchos de los productos de alta tecnología que nos rodean, de los motores de búsqueda de webs, del reconocimiento de habla de nuestros dispositivos, nos recomienda películas y series en nuestras plataformas de streaming favoritas, detecta el spam de nuestros correos, etc.

## Pero ¿qué es machine Learning y qué significa que una máquina pueda aprender algo? 

Según la definición académica de [Arthur Samuel](https://en.wikipedia.org/wiki/Machine_learning), que popularizó dicho término en 1959, *machine Learning es el campo de estudio que proporciona a los ordenadores la habilidad de aprender sin ser explícitamente programados*. Como informáticos que somos, aquí va otra definición más “ingenieril”: *Un programa de ordenador se dice que aprende de una experiencia E con respecto a alguna tarea T y alguna medida de la ejecución P, si su ejecución en T, medida por P, mejora con la experiencia E. (Tom Mitchell, 1997)*

En vista de esto ¿si nos descargamos una copia de [Wikipedia](https://es.wikipedia.org/) o la [Hemeroteca Digital](http://www.bne.es/es/Catalogos/HemerotecaDigital), nuestros ordenadores están aprendiendo algo? Evidentemente no. Dispondremos de una cantidad enorme de datos, pero de repente nuestras máquinas no serán mejores en ninguna tarea.

¿Qué ventajas ofrece el uso de machine learning sobre otras técnicas de programación tradicionales? Utilizando el caso de uso del spam de correo que mencionamos anteriormente, con un enfoque tradicional haríamos lo siguiente:

+ Observaríamos que en los correos de spam aparecen palabras del tipo “para ti”, “gratis”, “increíble”, etc.
  
+ Codificaríamos un procedimiento que detectara estas palabras y etiquetaríamos como spam aquellos correos que contuvieran estos patrones. 
  
+ Iteraríamos tantas veces por los dos pasos anteriores para codificar tantas reglas como patrones detectemos. 

Un enfoque basado en técnicas de machine learning se centraría en aprender qué palabras o frases aparecen con mayor frecuencia en correos etiquetados como spam en comparación con correos “buenos”. Es lo que se denomina “entrenar” nuestro modelo, con el objetivo de que pueda clasificar los nuevos correos que nos lleguen.

Además, supongamos que nuestro inteligente spammer compulsivo detecta que le bloqueamos aquellos correos donde aparece la palabra “gratis” y empieza a sustituirla por la palabra “gratuito”, y así sucesivamente cambiando las reglas. Un enfoque tradicional nos obligaría a estar constantemente cambiando nuestros patrones de detección y haciendo re-entregas. Un enfoque basado en ML detectaría automáticamente estos patrones inusualmente frecuentes en los correos marcados como spam y los marcaría en el futuro sin intervención humana.

Otro campo donde realmente brilla machine learning es en el reconocimiento de escritura manual (o del habla). Podríamos escribir un programa que detectara determinados trazos o incluso el alfabeto completo, pero esto no escalaría a los miles de combinaciones escritas por millones de personas en el mundo. La mejor forma sería entrenar un modelo de ML proporcionándole muchos ejemplos de diferentes tipos de letras y patrones escritos a mano.

Como vemos, machine learning es ideal para procesos donde tengamos mucho ajuste manual o un gran número de reglas, soluciones donde haya que adaptarse a nuevos datos, tratamiento de información no estructurada (sonidos, imágenes) y un largo etcétera de casos de uso.

## Clasificación de los sistemas de machine learning

Existen formas muy diversas de clasificar los sistemas de machine learning. Las más comunes serían las siguientes:

+ Si son entrenados con supervisión humana se pueden clasificar en: supervisados, no supervisados, semisupervisados y aprendizaje por reforzamiento.
 
+ Si pueden aprender incrementalmente al vuelo: aprendizaje online y aprendizaje por lotes.
  
+ Aprendizaje basado en instancia (donde los sistemas aprenden ejemplos “de memoria” y después generalizan a nuevos ejemplos usando medidas de similitud) vs aprendizaje basado en modelo (el sistema crea un modelo a partir de ejemplos de entrenamiento que usará posteriormente para realizar predicciones).
  
Esta tipología no es excluyente. Nuestro sistema de spam podría ser un ejemplo de aprendizaje supervisado online basado en modelo si lo entrenamos con una red neuronal.

Veamos un poco más cerca nuestra primera categorización. Una mañana cualquiera nos acercamos a nuestro "banco amigo" a pedir un préstamo para montar nuestro soñado puesto de castañas. Después de rellenar varios formularios con datos de todo tipo, el director de la sucursal nos convoca para la semana siguiente, donde nos comunicará si nos concede dicho préstamo. ¿Cómo sabe el banco si devolveremos el préstamo? El banco tiene información de otros cientos de miles de operaciones similares a la nuestra y conoce si el cliente devolvió el préstamo o no (es decir, tiene datos etiquetados, aprendizaje supervisado). Con los datos que les hemos proporcionado y con sus modelos de clasificación, el banco puede predecir con un nivel de probabilidad en qué medida seremos capaces de devolver el préstamo. Queda a criterio del director de la sucursal si confiar ciegamente en lo que pronostican dichos modelos.

En el aprendizaje no supervisado no disponemos de datos etiquetados, por lo que el sistema debe aprender sin contar con un profesor. Los algoritmos no supervisados son muy útiles para detectar relaciones o agrupaciones entre los datos, algo que a una persona le resultaría muy difícil detectar. Por ejemplo, los modelos detrás de las empresas de venta online pueden detectar que las personas que compran un determinado producto X también suelen comprar el producto Z, por lo que nos los suelen sugerir (“*Tal vez le interese…*”, “*Otros clientes también compraron…*”, etc.) durante el proceso de compra. Este tipo de algoritmos no supervisados también se usan para la detección de anomalías (muy útil en la prevención del fraude bancario o en la detección de defectos de fabricación). El sistema está entrenado con ejemplos normales, por lo que es capaz de determinar si una nueva instancia es o no una anomalía.

Algunos sistemas de clasificación de imágenes serían un ejemplo de aprendizaje semisupervisado: son capaces de detectar personas y probablemente determinará que la persona X aparece en el siguiente grupo de imágenes. Tan solo hay que ayudarle indicándole quién es esa persona para que a la siguiente ocasión sepa etiquetarla correctamente.

Por último, el aprendizaje por reforzamiento es un tipo muy diferente a los anteriores. El sistema obtiene recompensas o penalizaciones en función de sus acciones. Debe aprender a partir de ellas, eligiendo cuál sería la mejor estrategia (denominada *política*) para obtener la mayor recompensa a lo largo del tiempo. AlphaGo sería un ejemplo de aprendizaje por refuerzo. Aprendió su política ganadora estudiando millones de partidas. Durante su combate con el campeón del mundo aplicó las políticas que había aprendido.

En posteriores artículos hablaremos también de algunos de los lenguajes más idóneos y la combinación de herramientas que tenemos a nuestra disposición para trabajar de forma inmediata en machine learning: Python, Jupyter Notebook, Scikit-Learn, Tensor-Flow, Keras, etc.

Revisaremos cuáles son las fases principales de un proyecto típico de machine learning con ejemplos prácticos.

**¡Bienvenidos al mundo de machine learning!**

Por cierto, apenas 3 años después de su derrota por AlphaGo, Lee Sedol [se retiró de las competiciones oficiales](https://en.yna.co.kr/view/AEN20191127004800315). Si tenéis oportunidad no dejéis de ver el documental [AlphaGo – The movie](https://youtu.be/WXuK6gekU1Y), que narra la apasionante crónica del combate entre ambas “mentes”.

![go campeonato](/static/images/alphago-1024x576.jpg)