from typing import Union, List
import os
import string
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dateutil import parser
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


app = FastAPI()


#Función del incio
@app.get("/")
def read_root():
    welcome_message = 'Proyecto Individual - MLOps - maxilucchesi'
    return welcome_message



# Función UsersWorstDeveloper

@app.get("/worst_developers/{year}")
def get_worst_developers(year: int):
    """
    Endpoint para obtener el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para un año dado.

    :param year: Año para el cual se quiere obtener el top 3 de desarrolladoras con juegos menos recomendados.
    :type year: int
    :return: Lista de diccionarios con el top 3 de desarrolladoras.
    :rtype: list[dict]
    """
    return UsersWorstDeveloper(year)

def UsersWorstDeveloper(year: int):
    """
    Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado.

    :param year: Año para el cual se quiere obtener el top 3 de desarrolladoras con juegos menos recomendados.
    :type year: int
    :return: Lista de diccionarios con el top 3 de desarrolladoras.
    :rtype: list[dict]
    """

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'data', 'developer_negative_review.parquet')
    developer_negative_review = pq.read_table(path_to_parquet).to_pandas()

    # Filtra el dataframe para el año dado y sentiment_analysis es 0
    df_filtered = developer_negative_review[(developer_negative_review['year_review'] == year) &
                                            (developer_negative_review['sentiment_analysis'] == 0)]

    # Si no hay datos filtrados, retornar None
    if df_filtered.empty:
        return None

    # Agrupa por desarrollador y cuenta la cantidad de juegos recomendados
    peores_devs = df_filtered.groupby('developer')['recommend'].sum().nlargest(3)

    # Construye el resultado como una lista de diccionarios
    result = [{"Top 3{}".format(i + 1): developer} for i, (developer, _) in enumerate(peores_devs.items())]

    return result




# Función sentiment_analysis

@app.get("/sentiment-analysis/{dev}")
def sentiment_analysis(dev: str):

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'data', 'id_dev_sentiment.parquet')
    id_dev_sentiment = pq.read_table(path_to_parquet).to_pandas()
    

    # Filtra el dataframe para el desarrollador dado
    df_filtered = id_dev_sentiment[id_dev_sentiment['developer'] == dev]

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron registros para el desarrollador {dev}")

    # Agrupa por análisis de sentimiento y cuenta la cantidad de registros
    analysis_counts = df_filtered.groupby('sentiment_analysis').size().to_dict()

    # Construye el resultado como un diccionario con una lista
    result = {dev: [f'Negative = {analysis_counts.get(0, 0)}', f'Positive = {analysis_counts.get(2, 0)}']}

    return result







# Modelo de Recomendación de juegos, ingresando su id

@app.get("/recomendacion_juego/{id_juego}")
def recomenacion_juego(id_juego: int):

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'data', 'df_games_filtered.parquet')
    df_games_filtered = pq.read_table(path_to_parquet).to_pandas()
    df = df_games_filtered

    # Se instancia un objeto vectorizador usando TfidfVectorizer, con stop words en inglés.
    vectorizador = TfidfVectorizer(stop_words='english')

    # Los valores nulos en la columna 'tags_concat' se reemplazan con una cadena vacía.
    df_games_filtered['tags_concat'] = df_games_filtered['tags_concat'].fillna('')

    # Se realiza la transformación TF-IDF en los datos de la columna 'tags_concat'.
    vector_matrix = vectorizador.fit_transform(df_games_filtered['tags_concat'])

    # Se calcula la similitud del coseno
    simil_coseno = linear_kernel(vector_matrix, vector_matrix)

    # Se crea una serie de indices
    index = pd.Series(df_games_filtered.index, index=df_games_filtered['id']).drop_duplicates()

    try:
        # Buscar el nombre del juego correspondiente al ID
        nombre_juego = df_games_filtered.loc[df_games_filtered['id'] == id, 'app_name'].iloc[0]

        indice = index[id]
        similitud = list(enumerate(simil_coseno[indice]))
        similitud = sorted(similitud, key=lambda x: x[1], reverse=True)
        idx_game_recommended = [i[0] for i in similitud[1:6]]
        resultado_recomendacion = df_games_filtered['app_name'].iloc[idx_game_recommended]

        print('Si te gustó:', nombre_juego, '\n')
        print('Te recomendamos:')
        for juego_recomendado in resultado_recomendacion:
            print('-', juego_recomendado)

    except KeyError:
        print(id, 'no se encuentra en nuestra base.')