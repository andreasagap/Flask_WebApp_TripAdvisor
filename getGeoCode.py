import pandas as pd
from geopy import Nominatim
import numpy as np
import math

df = pd.read_csv('Analytics/demographics.csv')
print(df.isnull().sum())

print(df.head())
print(df.columns)

country_number = {}
# df['geocode'] = ''
geolocator = Nominatim(user_agent="Acropolis")
for i in range(len(df)):
    print(i)
    if i >= 1033:
        locationStr = df.iloc[i]['location']
        print(locationStr)
        print(type(locationStr))
        if not pd.isnull(locationStr):
            print('IN')
            location = geolocator.geocode(locationStr, timeout=10)
            if location:
                location = geolocator.reverse([location.latitude, location.longitude], timeout=10)
                try:
                    code = location.raw['address']['country_code'].upper()
                    print(code)
                    df.at[i, 'geocode'] = code
                    df.to_csv('Analytics/demographics.csv', index=False)
                except:
                    print('code not found')



