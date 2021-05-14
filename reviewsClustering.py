import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Concatenate review dataframes
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



# Topic modeling
# Vectorizing
count_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                   stop_words='english',
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6, max_features=4000)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                   stop_words='english',
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6, max_features=4000)

cv_data = count_vectorizer.fit_transform(reviews.text)
tfidf_data = tfidf_vectorizer.fit_transform(reviews.text)

# Dimensionality Reduction

n_comp = 10
lsa_tfidf = TruncatedSVD(n_components=n_comp)
lsa_cv = TruncatedSVD(n_components=n_comp)
nmf_tfidf = NMF(n_components=n_comp)
nmf_cv = NMF(n_components=n_comp)

lsa_tfidf_data = lsa_tfidf.fit_transform(tfidf_data)
lsa_cv_data = lsa_cv.fit_transform(cv_data)
nmf_tfidf_data = nmf_tfidf.fit_transform(tfidf_data)
nmf_cv_data = nmf_cv.fit_transform(cv_data)

# Scale data
scaler = StandardScaler()

lsa_tfidf_data_sc = scaler.fit_transform(lsa_tfidf_data)
lsa_cv_data_sc = scaler.fit_transform(lsa_cv_data)
nmf_tfidf_data_scaled = scaler.fit_transform(nmf_tfidf_data)
nmf_cv_data_scaled = scaler.fit_transform(nmf_cv_data)

# Cluster topics
kmeans = KMeans(n_clusters=5, random_state=0).fit(lsa_cv_data_sc)
embedded = TSNE(n_components=2).fit_transform(lsa_tfidf_data_sc)

# Visualize results
df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = embedded[:, 0]
df_subset['tsne-2d-two'] = embedded[:, 1]

plt.figure(figsize=(16, 8))
y = kmeans.labels_
df_subset['y'] = y
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue='y',
    palette=sns.color_palette("hls", kmeans.n_clusters),
    data=df_subset,
    legend="full",
    alpha=0.3
)


# Print most relevant reviews for each cluster
pd.options.display.max_colwidth = 90
for i in range(0, kmeans.n_clusters):
    print('---Cluster ' + str(i) + '---')
    indices_max = [index for index, value in enumerate(kmeans.labels_) if value == i]
    for rev_index in indices_max[:5]:
        print(rev_index, str(reviews.text[rev_index]))
        print("\n")

