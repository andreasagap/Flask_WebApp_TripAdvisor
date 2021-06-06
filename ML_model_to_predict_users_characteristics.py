#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import collections
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report
import re
import emojis
from sklearn import svm


def plotHeatMap(df):
    plt.figure(figsize=(16, 6))
    # calculate the correlation matrix
    corr = df.corr()

    # plot the heatmap
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()


def textPreprocessing(df):
    stop = stopwords.words('english')
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    return df


def TFIDFtransformer(reviewsDF):
    tfidf = TfidfVectorizer(max_features=200, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 3))
    tfidfDF = pd.DataFrame(tfidf.fit_transform(reviewsDF['text']).toarray())
    cols = tfidf.get_feature_names()
    # print(tfidf.get_feature_names())
    return tfidfDF, cols


def createAdditionalFeatures(df):

    # df['emojis'] = df['text'].apply(lambda x: len(re.findall(r'[\U0001f600-\U0001f650]', x)))
    # df['emojis'] = df['text'].apply(lambda x: emojis.count(x))
    # df['emojis'] = df['text'].apply(lambda x:  len(re.compile(r"\d+(.*?)[\u263a-\U0001f645]").findall(x)))
    df['hasKids'] = df.text.str.count("kid|kids")
    df['hasFamily'] = df.text.str.count("family|families")
    # df['wife'] = df.textstr.count("wife|husband")
    df['exclamationMarks'] = df['text'].apply(lambda x: len(re.findall(r'[!]', x)))
    df['questionMarks'] = df['text'].apply(lambda x: len(re.findall(r'[?]', x)))
    return df



def userFeatureExtraction(reviewsDF):
    # filter dataframe by username
    # userDF = reviewsDF[reviewsDF['username'] == username]
    # userDF['corpus'] = df.groupby(['username'])['text'].transform(lambda x : ' '.join(x))
    reviewsDF = reviewsDF.groupby('username').agg({
                          'placeType': lambda x: ",".join(x),
                          'reviewScore': 'mean',
                          'text': lambda x: ' '.join(x)
    }).reset_index()

    # add frequency per type of place
    reviewsDF = countFrequencies(reviewsDF)

    # add more features
    # count occurrences of specific words like 'kids', 'family

    # count occurrences of specific words like kids, family
    # emotion detection & topic detection
    reviewsDF = createAdditionalFeatures(reviewsDF)

    # text cleaning
    reviewsDF = textPreprocessing(reviewsDF)

    # vectorize with TFIDF
    tfidfDF, cols = TFIDFtransformer(reviewsDF)
    tfidfDF.columns = cols
    # print(type(cols))
    # print(cols)

    # concat the DataFrames
    df_concat = pd.concat([reviewsDF, tfidfDF], axis=1)
    print(df_concat.head(10))
    # print(df_concat.columns)

    df_concat = df_concat.drop(['placeType', 'text'], axis=1)

    return df_concat


def getUniquePlacesTypes(df):
    placeTypeList = reviewsDF['placeType'].tolist()
    allTypes = []
    for x in placeTypeList:
        x = x.split(",")
        for p in x:
            allTypes.append(p)

    uniqueTypes = list(set(allTypes))
    return uniqueTypes


def countFrequencies(df):

    types = getUniquePlacesTypes(df)
    for tt in types:
        df[tt] = 0.0

    for i in range(len(df)):
        pTypes = df.iloc[i]['placeType'].split(",")
        countReviews = len(pTypes)
        # print('count reviews: ', countReviews)
        ctr = dict(collections.Counter(pTypes))
        for key in ctr.keys():
            freq = ctr.get(key)
            # print('frequency: ', freq)
            df.at[i, key] = freq/countReviews

    return df


def keepOnlyUsersWithReviews(reviewsDF, df):
    # list of unique usernames
    usersWithReviews = list(set(reviewsDF['username'].tolist()))
    rdf = df[df['username'].isin(usersWithReviews)]
    return rdf


# Convert tags to list and One-Hot-Encoding
def getDummiesForTags(df):
    # convert to list
    df.tags = df.tags.apply(lambda y: np.nan if len(eval(y)) == 0 else eval(y))
    # one hot encoding
    tagsDummies = df['tags'].str.join('|').str.get_dummies()
    df = df.drop('tags', axis=1)
    df = df.join(tagsDummies)
    return df


def modelEvalutation(y_train, y_test, y_train_pred, y_test_pred):
    print('Train Accuracy: %.2f' % metrics.accuracy_score(y_train, y_train_pred))
    print('Train Balanced Accuracy: %.2f' % metrics.balanced_accuracy_score(y_train, y_train_pred))
    print('Train Recall: %.2f' % metrics.recall_score(y_train, y_train_pred, average='micro'))
    print('Train Precision: %.2f' % metrics.precision_score(y_train, y_train_pred, average='micro'))
    print('Train F1: %.2f' % metrics.f1_score(y_train, y_train_pred, average='micro'))

    print('Test Accuracy: %.2f' % metrics.accuracy_score(y_test, y_test_pred))
    print('Test Balanced Accuracy: %.2f' % metrics.balanced_accuracy_score(y_test, y_test_pred))
    print('Test Recall: %.2f' % metrics.recall_score(y_test, y_test_pred, average='micro'))
    print('Test Precision: %.2f' % metrics.precision_score(y_test, y_test_pred, average='micro'))
    print('Test F1: %.2f' % metrics.f1_score(y_test, y_test_pred, average='micro'))


def dropHighlyCorrelatedFeatures(df, threshold):
    print('Initial size: ', len(df.columns))
    df_corr = df.corr()
    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i + 1, df_corr.shape[0]):
            if abs(df_corr.iloc[i, j]) >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    print('New size: ', len(df.columns))

    return df

if __name__ == '__main__':

    # # Demographics DataFrame
    # df = pd.read_csv('/home/andreas/dav/tripadvisor/demographics.csv')
    #
    # print(df.columns)
    # # User Reviews DataFrame
    # reviewsDF = pd.read_csv('/home/andreas/Documents/Notebooks/TripAdvisor/reviews.csv')
    #
    # # get user attributes from their reviews
    # reviewsDF = userFeatureExtraction(reviewsDF)
    # # reviewsDF.to_csv('reviewsTFIDF.csv', index=False)
    # # reviewsDF = pd.read_csv('reviewsTFIDF.csv')
    #
    # print(reviewsDF.head(10))
    # print(reviewsDF.dtypes)
    #
    # # Count missing values per column
    # print(df.isnull().sum())
    # print(df.dtypes)
    #
    # # Drop rows without 'Age Group' info
    # df = df[df['age_group'].notna()]
    # # keep only users with reviews
    # df = keepOnlyUsersWithReviews(reviewsDF, df)
    # # one hot encoding for categorical features
    # print(df.dtypes)
    # df = getDummiesForTags(df)
    #
    # # Drop features irrelevant with 'Age'
    # df = df.drop(['gender', 'location'], axis=1)
    #
    # # count missing values per column and drop rows with missing values
    # df.isnull().sum()
    # df = df.dropna()
    # df.isnull().sum()
    #
    # # count frequencies per class
    # print(df['age_group'].value_counts())
    # # dataset size
    # print('df demographics:', len(df))
    #
    # # integration of 2 DataFrames
    # mergeDF = pd.merge(df, reviewsDF, on=['username'])
    # print(mergeDF.head(10))
    # print('final df:', len(mergeDF))
    #
    # print(mergeDF.dtypes)
    # print(mergeDF.columns)
    #
    # # drop username
    # mergeDF = mergeDF.drop(['username'], axis=1)
    # # print columns
    # for att in mergeDF.columns:
    #     print(att)
    #
    # df = mergeDF

    # df.to_csv('integratedDF.csv', index=False)
    df = pd.read_csv('integratedDF.csv')

    # only one case - delete to avoid error in stratification
    df = df[df['age_group'] != '13-17']

    # df = df.iloc[:, :-200]

    # plotHeatMap(df)
    # correlation with ta
    corrDF = df.drop(["age_group"], axis=1).apply(lambda x: x.corr(df.age_group.astype('category').cat.codes))
    sortedCorrDF = corrDF.abs().sort_values(ascending=False)
    print(sortedCorrDF.head(10))

    # df = dropHighlyCorrelatedFeatures(df, 0.9)

    X = df.drop('age_group', axis=1)
    y = df[['age_group']]


    # Train-Test Split # stratify=y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Min Max Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE for Imbalance Learning
    # strategy = {'18-24': 100, '25-34': 200, '35-49': 200}
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    print(len(X_train))
    undersample = TomekLinks()
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    print(len(X_train))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Model Training
    from sklearn.linear_model import LogisticRegression

    # clf = RandomForestClassifier(max_depth=3)
    clf = svm.SVC(decision_function_shape='ovo')
    # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train, y_train)

    # plot confusion matrix
    target_names = list(set(df['age_group'].to_list()))
    plot_confusion_matrix(clf, X_test, y_test,
                          cmap=plt.cm.Blues,
                          normalize=None)
    plt.show()

    # Predictions for train/test set
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # classification report
    clf_report = classification_report(y_test, y_test_pred)
    print(clf_report)

    # Evaluation (print Accuracy, Precision, Recall, F1
    modelEvalutation(y_train, y_test, y_train_pred, y_test_pred)
