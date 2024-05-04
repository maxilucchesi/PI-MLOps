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
    welcome_message =  '¡Bienvenidos al Sistema de Recomendación de Juegos de Steam! Vas a poder recibir recomendaciones con base en tus juegos favoritos. Además, podrás realizar otro tipo de consultas que te ayudarán a elegir mejor tu próximo juego. ¡Explora nuestro sistema y descubre los juegos que te encantarán! --> Agregar /docs al final de la URL para poder comenzar :) Autor: Maximiliano Lucchesi - Data Scientist Jr @ Steam'
    return welcome_message



# Función UsersWorstDeveloper

@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year: int):

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'archivos', 'game_negative_review.parquet')
    game_negative_review = pq.read_table(path_to_parquet).to_pandas()

    # Filtra el dataframe para el año dado y sentiment_analysis es 0
    df_filtered = game_negative_review[(game_negative_review['year_review'] == year) &
                                            (game_negative_review['sentiment_analysis'] == 0)]

    # Si no hay datos filtrados, retornar None
    if df_filtered.empty:
        return None

    # Agrupa por juego y cuenta la cantidad de juegos recomendados
    peores_games = df_filtered.groupby('app_name')['recommend'].sum().nlargest(3)

    # Construye el resultado como una lista de diccionarios
    result = [{"Top {}".format(i + 1): app_name} for i, (app_name, _) in enumerate(peores_games.items())]

    return result



# Función sentiment_analysis

@app.get("/sentiment-analysis/{year}")
def sentiment_analysis(year: int):

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'archivos', 'year_sentiment.parquet')
    year_sentiment = pq.read_table(path_to_parquet).to_pandas()
    

    # Filtra el dataframe para el desarrollador dado
    df_filtered = year_sentiment[year_sentiment['year_developed'] == year]

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron registros para el desarrollador {year}")

    # Agrupa por análisis de sentimiento y cuenta la cantidad de registros
    analysis_counts = df_filtered.groupby('sentiment_analysis').size().to_dict()

    # Construye el resultado como un diccionario con una lista
    result = [f'Negative = {analysis_counts.get(0, 0)}', f'Positive = {analysis_counts.get(2, 0)}']

    return result



# Modelo de Recomendación de juegos, ingresando su id

@app.get("/recomendacion_juego/{id}")
def recomendacion_juego(id: int):

    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'archivos', 'df_games_filtered.parquet')
    df_games_filtered = pq.read_table(path_to_parquet).to_pandas()

    vectorizador = TfidfVectorizer(stop_words='english')
    df_games_filtered['tags_concat'] = df_games_filtered['tags_concat'].fillna('')
    vector_matrix = vectorizador.fit_transform(df_games_filtered['tags_concat'])
    simil_coseno = linear_kernel(vector_matrix, vector_matrix)
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