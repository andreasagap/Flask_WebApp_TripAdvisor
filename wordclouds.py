import pandas as pd
import numpy as np
import nltk
import re
import matplotlib
from nltk.corpus import stopwords
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.express as px
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
import datetime
import io
import urllib, base64


def load_dataset(start,end):
    reviews = pd.read_csv('Analytics/acropolis_reviews.csv',parse_dates=True)
    reviews['date'] = reviews['date'].apply(lambda v: datetime.datetime.strptime(v, '%Y-%m-%d'))
    #mask = (reviews['date'] >= start) & (reviews['date'] <= end)
   # prc_sub = reviews.loc[mask]
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    reviews = reviews.loc[(reviews['date'] > start_date) & (reviews['date'] < end_date)]
    print(reviews.dtypes)
    print(reviews['date'].min(), reviews['date'].max())
    print('Reviews shape: ', reviews.shape)
    return reviews

def create_histogram(reviews):
    fig = px.histogram(reviews, x="rating")
    fig.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5)
    fig.update_layout(title_text='Rating')
    fig.show()

def preprocessing(reviews):
    # Create stopword list:
    stopwords_ = set(stopwords.words('english'))
    stopwords_.update(["br", "href"])

    # assign reviews with score > 3 as positive sentiment
    # score < 3 negative sentiment
    reviews = reviews[reviews['rating'] != 3]

    reviews['sentiment'] = reviews['rating'].apply(lambda rating: +1 if rating > 3 else -1)

    # split reviews - positive and negative sentiment:
    positive = reviews[reviews['sentiment'] == 1]
    negative = reviews[reviews['sentiment'] == -1]
    print('so far so good')
    createWordcloud(positive, stopwords_)
    createWordcloud(negative, stopwords_)

    reviews['review_text'] = reviews['review_text'].apply(remove_punctuation)

    # keep only review text and rating
    dfNew = reviews[['review_text', 'rating']]
    dfNew.head()

    index = reviews.index
    reviews['random_number'] = np.random.randn(len(index))
    train = reviews[reviews['random_number'] <= 0.8]
    test = reviews[reviews['random_number'] > 0.8]

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['review_text'])
    test_matrix = vectorizer.transform(test['review_text'])
    return train_matrix, test_matrix, train, test


def createWordcloud(text, stopwords):
    text = " ".join(review for review in text.review_title)
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.savefig('static/img/wordcloud.png', transparent=True)
    return  wordcloud.words_
    #return True


    #plt.show()


def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"'))
    return final

def apiWordCloud(start,end):
    reviews = load_dataset(start,end)
    if len(reviews)==0:
        print(len(reviews))
        return False
    # Create stopword list:
    stopwords_ = set(stopwords.words('english'))
    stopwords_.update(["br", "href"])
    stopwords_.add("Athens")
    stopwords_.add("Acropolis")

    # assign reviews with score > 3 as positive sentiment
    # score < 3 negative sentiment
    reviews = reviews[reviews['rating'] != 3]

    reviews['sentiment'] = reviews['rating'].apply(lambda rating: +1 if rating > 3 else -1)

    # split reviews - positive and negative sentiment:
    positive = reviews[reviews['sentiment'] == 1]
    negative = reviews[reviews['sentiment'] == -1]
    return True,createWordcloud(positive, stopwords_),createWordcloud(negative, stopwords_)
    #createWordcloud(negative, stopwords_)

if __name__ == "__main__":
    reviews = load_dataset()
    create_histogram(reviews)
    train_matrix, test_matrix, train, test = preprocessing(reviews)




