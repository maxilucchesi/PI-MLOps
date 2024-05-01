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

@app.get("/")
def read_root():
    welcome_message = 'Proyecto Individual - MLOps - maxilucchesi'
    return welcome_message

def UsersWorstDeveloper(year: int):
    """
    Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado.

    :param year: Año para el cual se quiere obtener el top 3 de desarrolladoras con juegos menos recomendados.
    :type year: int
    :return: Lista de diccionarios con el top 3 de desarrolladoras.
    :rtype: list[dict]
    """

    # Obtener el directorio actual del archivo y construir la ruta al archivo parquet
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'data', 'developer_negative_review.parquet')

    # Leer el archivo parquet como DataFrame de pandas
    developer_negative_review = pq.read_table(path_to_parquet).to_pandas()

    # Filtrar el DataFrame por año y por comentarios negativos, que son aquellos que tienen valor 0 en 'sentiment_analysis'
    df_filtered = developer_negative_review[(developer_negative_review['year_review'] == year) &
                                            (developer_negative_review['sentiment_analysis'] == 0)]

    # Si no hay datos filtrados, retornar None
    if df_filtered.empty:
        return None

    # Agrupar por desarrolladora y sumar las recomendaciones negativas, luego obtener las 3 desarrolladoras con más recomendaciones negativas
    peores_devs = df_filtered.groupby('developer')['recommend'].sum().nlargest(3)

    # Crear la lista de diccionarios con el top 3 de desarrolladoras
    result = [{"Top 3{}".format(i + 1): developer} for i, (developer, _) in enumerate(peores_devs.items())]

    return result