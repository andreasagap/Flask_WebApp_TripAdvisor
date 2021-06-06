import json
from random import randrange
import random

import numpy
import pandas as pd
from geopy import Nominatim
import numpy as np
import math
# some JSON:
jsonD = json.loads('{"US": { "lat": 43.000000,"long": -75.000000},"GB": { "lat": 51.509865,"long": -0.118092},"CA": { "lat": 43.651070,"long": -79.347015}}')

# df = pd.read_csv('Analytics/demographics.csv')
# print(df.isnull().sum())
#
# print(df.head())
# print(df.columns)
#
# country_number = {}
# # df['geocode'] = ''
# geolocator = Nominatim(user_agent="Acropolis")
# for i in range(len(df)):
#     print(i)
#     if i >= 1033:
#         locationStr = df.iloc[i]['location']
#         print(locationStr)
#         print(type(locationStr))
#         if not pd.isnull(locationStr):
#             print('IN')
#             location = geolocator.geocode(locationStr, timeout=10)
#             if location:
#                 location = geolocator.reverse([location.latitude, location.longitude], timeout=10)
#                 try:
#                     code = location.raw['address']['country_code'].upper()
#                     print(code)
#                     df.at[i, 'geocode'] = code
#                     df.to_csv('Analytics/demographics.csv', index=False)
#                 except:
#                     print('code not found')
import geopy.distance
def getAirplanes(codeCountry):
    df = pd.read_csv("Analytics/ratings.csv")
    df = df.loc[df['code'] == codeCountry]
    df.dropna(subset=["latitude"], inplace=True)
    df = df[df.latitude != 0.0].drop_duplicates('placeLocation')
    df = df.sample(n=50)
    arrayTemp = []
    for i in range(len(df)):
        coords_1 = (df.iloc[i]['latitude'],df.iloc[i]['longitude'])
        coords_2 = (jsonD[codeCountry]["lat"], jsonD[codeCountry]["long"])
        if(geopy.distance.distance(coords_1, coords_2).km>2500):
            arrayTemp.append({
                "name": 'V131',
                "origin": {
                    "city": codeCountry,
                    "latitude": jsonD[codeCountry]["lat"],
                    "longitude": jsonD[codeCountry]["long"]
                },
                "destination": {
                    "city": df.iloc[i]['placeLocation'],
                    "latitude": df.iloc[i]['latitude'],
                    "longitude": df.iloc[i]['longitude']
                },
                "state": 1,
                "color": "%06x" % random.randint(0, 0xFFFFFF)
            })
    return json.dumps(arrayTemp)
def square(x,dfDemographs):
    return dfDemographs["geocode"]
def removeColumns():
    df = pd.read_csv("Analytics/ratings.csv")
    dfDemographs = pd.read_csv("Analytics/demographics.csv")
    df = df.drop('text', axis=1)
    df = df.drop('date', axis=1)
    df = df.drop('place', axis=1)
    df = df.drop('placeURL', axis=1)
    df = df.drop('placeType', axis=1)
    df = df.drop('reviewScore', axis=1)
    df = df.drop("avgScore",axis=1)
    df = df.drop('title', axis=1)
    dfDemographs = dfDemographs[dfDemographs.geocode.notnull()]
    print(dfDemographs)
    for i in range(len(df)):
        code = dfDemographs.loc[dfDemographs['username'] == df.iloc[i]['username']]
        print(code)
        if(len(code)):
            df.at[i, 'code'] = code.iloc[0].geocode

    df.to_csv('Analytics/ratingsNew.csv', index=False)
if __name__ == '__main__':

    import pandas as pd

    geolocator = Nominatim(user_agent="Acropolis")
    df = pd.read_csv('Analytics/ratingsNew.csv').loc[3000:4000]

    from geopy.extra.rate_limiter import RateLimiter

    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    from tqdm import tqdm

    tqdm.pandas()
    df['location'] = df['placeLocation'].progress_apply(geocode)
    df['latitude'] = df['location'].apply(lambda loc: loc.point.latitude if loc else None)
    df['longitude'] = df['location'].apply(lambda loc: loc.point.longitude if loc else None)
    df.to_csv('Analytics/ratings.csv',mode='a', index=False)
    # frame = pd.concat(li, axis=0, ignore_index=True)
    # frame['latitude'] = 0.0
    # frame['longitude'] = 0.0
    # for i in range(len(frame)):
    #     print(i)
    #       if i >= 400:
    #         locationStr = frame.iloc[i]['placeLocation']
    #         location = geolocator.geocode(locationStr, timeout=10)
    #         print(location.latitude)
    #         if location:
    #             frame.at[i, 'latitude'] = location.latitude
    #             frame.at[i, 'longitude'] = location.longitude
    #             frame.to_csv('Analytics/ratings.csv', index=False)

