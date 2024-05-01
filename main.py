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

    # Lee el archivo parquet y obtiene la ruta del directorio actual del script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_parquet = os.path.join(current_directory, 'data', 'developer_negative_review.parquet')
    developer_negative_review = pq.read_table(path_to_parquet).to_pandas()

    # Filtra el dataframe para el año dado y donde recommend es True y sentiment_analysis es positivo
    df_filtered = developer_negative_review[(developer_negative_review['year_review'] == year) &
                                               (developer_negative_review['sentiment_analysis'] == 0)]

    if df_filtered.empty:
        return None

    # Agrupa por desarrollador y cuenta la cantidad de juegos recomendados
    peores_devs = df_filtered.groupby('developer')['recommend'].sum().nlargest(3)

    # Construye el resultado como una lista de diccionarios
    result = [{"Top 3{}".format(i + 1): developer} for i, (developer, _) in enumerate(peores_devs.items())]

    return result