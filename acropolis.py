import pandas as pd



def getAcropolisStatistics():
    data = pd.read_csv("Analytics/demographics.csv")
    ratings_acropolis = len(data)
    gender = data.gender.str.lower().value_counts()
    ages = data.age_group.value_counts()
    print(ages)
    return ratings_acropolis,gender["man"],gender["woman"],ages