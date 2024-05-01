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

# Función UsersWorstDeveloper
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