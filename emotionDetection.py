import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def loadDataset():
    # Import reviews and concatenate
    reviews0 = pd.read_csv('Analytics/reviews/reviews0.csv')
    reviews1 = pd.read_csv('Analytics/reviews/reviews1.csv')
    reviews2 = pd.read_csv('Analytics/reviews/reviews2.csv')
    reviews3 = pd.read_csv('Analytics/reviews/reviews3.csv')
    reviews4 = pd.read_csv('Analytics/reviews/reviews4.csv')
    reviews5 = pd.read_csv('Analytics/reviews/reviews5.csv')
    reviews6 = pd.read_csv('Analytics/reviews/reviews6.csv')
    reviews7 = pd.read_csv('Analytics/reviews/reviews7.csv')
    reviews = [reviews0, reviews1, reviews2, reviews3, reviews4, reviews5, reviews6, reviews7]
    reviews = pd.concat(reviews, ignore_index=True)
    return reviews


if __name__ == "main":
    reviews = loadDataset()

    # group reviews by place location
    placeLocation = reviews.groupby('placeLocation')
    locationsCount = placeLocation.size()
    locationsCount.sort_values(inplace=True, ascending=False)
    print(locationsCount.head(50))

    # take top 10 locations according to review count
    top10locations = locationsCount.head(11).index
    # remove world location
    top10locations = top10locations.drop('World')

    for location in top10locations:
        # select each location reviews
        reviewsLocation = pd.DataFrame(reviews['text'].loc[reviews['placeLocation'] == location])
        # initialize vader analyzer
        sid = SentimentIntensityAnalyzer()
        # compute score for each review and store them in scores column
        reviewsLocation['scores'] = reviewsLocation['text'].apply(lambda review: sid.polarity_scores(review))

        reviewsLocation['compound'] = reviewsLocation['scores'].apply(lambda score_dict: score_dict['compound'])

        scores = reviewsLocation['compound'].values
        np.savetxt("Sentiment Scores/" + str(location) + ".csv", scores, delimiter=",")

        # histogram of reviews scores per location
        bins = [-1, -0.5, 0, 0.5, 1]
        plt.figure()
        plt.hist(scores, bins, histtype='bar', rwidth=0.8)
        plt.title('VADER sentiment scores for %s' % location)
        filename = location.replace(", ", "")
        plt.pause(0.05)
        plt.savefig('Vader/%s.png' % filename)
