import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import seaborn as sns


def textPreprocessing(df):
    stop = stopwords.words('english')
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    stemmer = PorterStemmer()

    df['text'] = df['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    return df


def TFIDFtransformer(df):
    tfidf = TfidfVectorizer(max_features=100, analyzer='word', ngram_range=(1, 3), stop_words='english')
    tfidfDF = pd.DataFrame(tfidf.fit_transform(df['text']).toarray())
    cols = tfidf.get_feature_names()
    return tfidfDF, cols


# elbow rule plot
def optimalK_SSE(data):
    sum_of_squared_distances = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def optimalK_Silouhette(data):
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k)
        cluster_labels = km.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('Silhouette For Optimal k')
    plt.show()


def plotSilouhette(X):
    range_n_clusters = range(2, 11)

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

    plt.show()


def intepretationWithVariance():

    print()


if __name__ == '__main__':

    # load reviews csv
    # df = pd.read_csv('/home/andreas/Documents/Notebooks/TripAdvisor/reviews.csv')
    # print(df.head())
    # print(df.dtypes)
    #
    # # keep ony text
    # df = df[['username', 'text']]
    #
    # # group by username & merge reviews text to a unified corpus
    # df = df.groupby('username').agg({
    #     'text': lambda x: ' '.join(x)
    # }).reset_index()
    #
    # # Text Pre-Processing
    # df = textPreprocessing(df)
    #
    # # Text Vectorization using TFIDF
    # df, cols = TFIDFtransformer(df)
    # df.columns = cols
    # print(df.head())
    #
    # df.to_csv('tfidf.csv', index=False)
    df = pd.read_csv('tfidf.csv')

    # print('before outliers:', len(df))
    # remove outliers
    # from scipy import stats
    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # print('after outliers:', len(df))

    pca = PCA(n_components=2)
    pcadf = pd.DataFrame(pca.fit_transform(df))

    print(pca.explained_variance_ratio_)

    # df.to_csv('pcaDF.csv', index=False)

    # pcadf = pd.read_csv('pcaDF.csv')

    # Clustering Evaluation
    optimalK_SSE(pcadf)
    optimalK_Silouhette(pcadf)
    # plotSilouhette(pcadf)


    # dbscan = DBSCAN(eps=1, min_samples=2).fit(df)
    # print('DBSCAN: {}'.format(silhouette_score(df, dbscan.labels_,
    #                                            metric='cosine')))

    km = KMeans(n_clusters=3).fit(pcadf)
    # cluster_labels = km.fit_predict(df)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df))
    df_scaled['cluster_id'] = km.labels_
    df_mean = df_scaled.groupby('cluster_id').mean()
    # df_mean.columns  = df.columns

    results = pd.DataFrame(columns=['Variable', 'Var'])
    for column in df_mean.columns:
        print(column)
        results.loc[len(results), :] = [column, np.var(df_mean[column])]
    selected_columns = list(results.sort_values(
        'Var', ascending=False,
    ).head(15).Variable.values) + ['cluster_id']
    tidy = df_scaled[selected_columns].melt(id_vars='cluster_id')
    tidy['variable'] = tidy['variable'].apply(lambda x: df.columns[x])
    # clrs = ['grey' if (x < max(tidy['value'])) else 'red' for x in tidy['value']]
    sns.barplot(x='cluster_id', y='value', hue='variable', data=tidy)
    plt.legend(bbox_to_anchor=(1.01, 1),
               borderaxespad=0)
    plt.title('Interpretation with feature variance')
    plt.show()

    # for i in selected_columns:
    #     print(str(i) + ': ' + str(df.columns[i]))


    from sklearn.ensemble import RandomForestClassifier

    X, y = df_scaled.iloc[:, :-1], df_scaled.iloc[:, -1]
    clf = RandomForestClassifier(n_estimators=100).fit(X, y)
    data = np.array([clf.feature_importances_, X.columns]).T
    columns = list(pd.DataFrame(data, columns=['Importance', 'Feature'])
                   .sort_values("Importance", ascending=False)
                   .head(15).Feature.values)
    tidy = df_scaled[columns + ['cluster_id']].melt(id_vars='cluster_id')
    tidy['variable'] = tidy['variable'].apply(lambda x: df.columns[x])
    sns.barplot(x='cluster_id', y='value', hue='variable', data=tidy)
    plt.legend(bbox_to_anchor=(1.01, 1),
               borderaxespad=0)
    plt.title('Interpretation with feature importance')
    plt.show()




