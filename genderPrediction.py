import pandas as pd
import os
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

users = pd.read_csv('Analytics\demographics.csv')

users = users.drop(columns=['member_since', 'cities', 'contributions', 'level', 'stars1', 'stars2',
                            'stars3', 'stars4', 'stars5'])
X = users['username']
y = users['gender']

y_train = y.dropna()
y_test = y[y.isnull()]
x_train = X.iloc[y_train.index]
x_test = X.iloc[y_test.index]

# Convert to lowercase
y_train = y_train.str.lower()

x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# Convert to binary
enc = LabelEncoder()
y_train = enc.fit_transform(y_train)

vectorizer = CountVectorizer(analyzer='char')
x_train = vectorizer.fit_transform(x_train.ravel()).toarray()
x_test = vectorizer.fit_transform(x_test.ravel()).toarray()

# Split to training and test set
X_train = x_train[:1500]
X_test = x_train[1501:]
Y_train = y_train[:1500]
Y_test = y_train[1501:]

# Classification
clf = LogisticRegression()
clf.fit(X_train, Y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print('Training accuracy:', accuracy_score(Y_train, y_pred_train))
print('Test accuracy:', accuracy_score(Y_test, y_pred_test))


