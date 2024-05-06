![Banner Henry](images/banner_henry_amarillo.png)

# 🚀 Proyecto Individual 1 🚀

## ⚙️ Machine Learning Operations ⚙️

## 🧪 Labs - DS PT 08 🧪
## Autor: Maximiliano Lucchesi 👨🏻‍💻

#### Propuesta de trabajo:

Se plantea un escenario en donde la empresa Steam nos contrata para realizar un trabajo de Data Science.

![Banner Steam](images/banner_steam.jpg)
**Sobre Steam:** Es una plataforma digital que vende, actualiza y distribuye videojuegos para computadoras, además de ofrecer funciones sociales y de gestión de juegos.

Nos proveen de la información con tres archivos JSON junto con un diccionario de datos, y a partir de ahí, se solicita construir el trabajo usando Python.

- `user_reviews.json`: tiene los comentarios que dejan los usuarios a los juegos, junto con información como: id de usuario, id de juego, fecha de posteo, si lo recomienda, etc.
- `users_items.json`: tiene la información de cuánto tiempo juegan los usuarios a cada juego que tienen, también indicando id de item, tiempo jugado, etc.
- `steam_games.json`: tiene toda la información de cada juego disponible en la plataforma: desarrollador, fecha de publicación, id, nombre, género, etc.

**La forma en que se abordó el proyecto fue la siguiente:**

1. Realizar un EDA inicial de reconocimiento.
2. Proceso de ETL.
3. Desarrollar un NLP para uno de los dataframes.
4. API y deployment.
5. Modelo de Machine Learning.

**La siguiente explicación también puede ser visualizada a través del siguiente video explicativo, donde se muestra paso a paso el trabajo realizado y se pone a prueba su funcionamiento.**

## EDA Inicial

En un primer reconocimiento, se pudo observar que algunas columnas de datos estaban anidadas, por lo cual, el primer paso fue usar la función 'normalize' para poder corregir esto y mergear el contenido de esta columna al resto del dataframe.

También, se hizo un análisis de los tipos de datos en cada uno de los dataframes, así también como los porcentajes de valores nulos con los que vinieron.

Este trabajo se puede visualizar en el archivo '`EDA.ipynb`' del repositorio.

## Extract, Transform and Load (ETL)

En esta parte, se realizó un proceso de transformación y limpieza de dataframes, conforme a la información relevada en la etapa anterior.

Dependiendo de cada archivo, se realizaron tratamientos de: eliminación de registros completamente nulos, modificación de tipos de datos para conveniencia de las consignas, registros vacíos, estandarización de ciertas columnas que contaban con varios tipos de datos, entre otros.

Luego, se fueron exportando nuevamente estos archivos para poder utilizar como base estos mismos. Muchos, al ser depurados, disminuyeron su tamaño brindando optimización al proyecto.

Este trabajo se puede visualizar en el archivo '`ProcesoETL.ipynb`' del repositorio.

## Natural Language Processing

En esta etapa, como pedía la consigna, se realizó un proceso de NLP para poder conocer el "sentimiento" de cada review hecha por el usuario. La idea es que este proceso pueda leer e interpretar el comentario y darle una polaridad al mismo. Siendo:

- negativo: se le asigna el valor '0'
- nulo, neutro, o sin comentario: se le asigna el valor '1'
- positivo: se le asigna el valor '2'

El resultado de este análisis, reemplaza la columna 'reviews' por 'sentiment_analysis'

Este trabajo se puede visualizar en el archivo '`natural_language_processing.ipynb`' del repositorio.

## API

Siguiendo la recomendación de la consigna, para hacer el deploy de este proyecto se utilizó el framework Render y FastAPI.

Fue necesario crear una cuenta y configurar el entorno virtual en Render para poder hacer la conexión con nuestro repositorio público de GitHub. Una vez configurado, se probó la conexión con unas líneas del `main.py`, imprimiendo "Hello World!".

Mediante [este link](https://pi-mlops-maxilucchesi.onrender.com), se puede acceder a la API, y con [este otro](https://pi-mlops-maxilucchesi.onrender.com/docs#/) ir directamente al entorno para realizar las consultas con los endpoints. Las cuales son las siguientes:

- `UsersNotRecommend(year: int)`: Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
- `sentiment_analysis(empresa_desarrolladora: str)`: Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

La conexión con la API fue mediante el archivo '`main.py`' de este repositorio.

## Modelo de Machine Learning

Para poder hacer el modelo de Machine Learning, se hizo un segundo EDA, orientado a poder identificar patrones útiles y relaciones claves. El objetivo también es poder visualizar mediante gráficos, variables interesantes de cada dataframe.

Por último, se desarrolló un Modelo de Recomendación para los usuarios, que consta de lo siguiente: ingresando el id del juego, uno puede recibir 5 juegos recomendados similares a ese.

Para lograr esto, se desarrolló un modelo de similitud del coseno que resultó muy útil y efectivo a la hora de ayudar a los usuarios a decidir su próximo juego.

En el archivo '`modelos_recomendacion.ipynb`' se puede encontrar el desarrollo completo del mismo.