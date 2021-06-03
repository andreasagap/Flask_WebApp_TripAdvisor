import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def textPreprocessing(df):
    stop = stopwords.words('english')
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    return df


def TFIDFtransformer(df):
    tfidf = TfidfVectorizer(max_features=500, analyzer='word', ngram_range=(1, 2))
    tfidfDF = pd.DataFrame(tfidf.fit_transform(df['text']).toarray())
    cols = tfidf.get_feature_names()
    return tfidfDF, cols


def loadDataset():
    reviews0 = pd.read_csv('Analytics/reviews/reviews0.csv')
    reviews1 = pd.read_csv('Analytics/reviews/reviews1.csv')
    reviews2 = pd.read_csv('Analytics/reviews/reviews2.csv')
    reviews3 = pd.read_csv('Analytics/reviews/reviews3.csv')
    reviews4 = pd.read_csv('Analytics/reviews/reviews4.csv')
    reviews5 = pd.read_csv('Analytics/reviews/reviews5.csv')
    reviews6 = pd.read_csv('Analytics/reviews/reviews6.csv')
    reviews7 = pd.read_csv('Analytics/reviews/reviews7.csv')
    reviews = [reviews0, reviews1, reviews2, reviews3, reviews4, reviews5, reviews6, reviews7]
    df = pd.concat(reviews, ignore_index=True)
    return df

if __name__ == 'main':
    # Load dataset
    df = loadDataset()

    # Keep text and username
    df = df[['username', 'text']]

    # group by username & merge reviews text to a unified corpus
    df = df.groupby('username').agg({
        'text': lambda x: ' '.join(x)
    }).reset_index()

    # Text Pre-Processing
    df = textPreprocessing(df)

    # Import demographics dataset
    demographics = pd.read_csv('Analytics/demographics.csv')

    # Keep only username and gender
    demographics = demographics[['username', 'gender']]

    # Merge the two dataframes
    dataframe = pd.merge(df, demographics, on='username')

    # X: text, y: gender
    X = dataframe['text']
    y = dataframe['gender']

    # Save nan instances and keep non nan for training and testing
    y_train = y.dropna()
    y_test = y[y.isnull()]
    x_train = X.iloc[y_train.index]
    x_test = X.iloc[y_test.index]

    # Convert target values to lowercase
    y_train = y_train.str.lower()

    x_train = x_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    # Encode terget values to labels
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)

    # Convert to vector representation
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train.ravel()).toarray()
    x_test = vectorizer.fit_transform(x_test.ravel()).toarray()

    # Split to training and test set
    X_train = x_train[:600]
    X_test = x_train[601:]
    Y_train = y_train[:600]
    Y_test = y_train[601:]

    # Initialize Random Forest Classifier and training phase
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    # Prediction phase
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Print results
    print('Training accuracy:', accuracy_score(Y_train, y_pred_train))
    print('Test accuracy:', accuracy_score(Y_test, y_pred_test))
    print('Test F1:', f1_score(Y_test, y_pred_test))

    # Print confusion matrix
    cm = confusion_matrix(Y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Man', 'Woman'], yticklabels=['Man', 'Woman'], cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Gender Prediction")
    plt.show()

