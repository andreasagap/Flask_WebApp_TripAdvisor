import json

import pandas as pd
from geopy.geocoders import Nominatim


def getAcropolisStatistics():
    data = pd.read_csv("Analytics/demographics_old.csv")
    ratings_acropolis = len(data)
    gender = data.gender.str.lower().value_counts()
    ages = data.age_group.value_counts()

    return ratings_acropolis, gender["man"], gender["woman"], ages


def getAcropolisMap():
    data = pd.read_csv("Analytics/demographics.csv")
    dictionary = {}
    for i, v in data.geocode.value_counts().items():
        dictionary[i] = v
    #a = pd.read_json(data.geocode.value_counts(), typ='series', orient='records')
    return json.dumps(dictionary)
