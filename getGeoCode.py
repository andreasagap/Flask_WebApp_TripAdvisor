import numpy
import pandas as pd
from geopy import Nominatim
import numpy as np
import math

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
    removeColumns()
    # import pandas as pd
    # import glob
    #
    # geolocator = Nominatim(user_agent="Acropolis")
    # #df = pd.read_csv('Analytics/demographics.csv')
    # path = r'/Analytics/reviews'  # use your path
    # all_files = glob.glob("Analytics/reviews/*.csv")
    #
    # li = []
    #
    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col=None, header=0)
    #     li.append(df)
    #
    # frame = pd.concat(li, axis=0, ignore_index=True)
    # frame['latitude'] = 0.0
    # frame['longitude'] = 0.0
    # for i in range(len(frame)):
    #     print(i)
          #if i >= 400:
        #     locationStr = frame.iloc[i]['placeLocation']
        #     location = geolocator.geocode(locationStr, timeout=10)
        #     print(location.latitude)
        #     if location:
        #         frame.at[i, 'latitude'] = location.latitude
        #         frame.at[i, 'longitude'] = location.longitude
        #         frame.to_csv('Analytics/ratings.csv', index=False)

