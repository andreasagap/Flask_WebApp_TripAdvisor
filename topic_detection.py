import pandas as pd
import numpy as np
import sklearn
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import pyLDAvis.gensim
import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models
from itertools import chain
import warnings
warnings.simplefilter('ignore')


data = pd.read_csv("Analytics/acropolis_reviews.csv")

#clean data
#create stopword list
stop = set(stopwords.words('english'))
new_stopwords = ['athens', 'acropolis', 'u']
stop = stop.union(new_stopwords)
#remove punctuation
exclude = set(string.punctuation)
#lemmatize
lemma = WordNetLemmatizer()

def clean(text):
    stop_free = ' '.join([word for word in text.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
    return normalized.split()

data['review_text_clean'] = data['review_text'].apply(clean)


# create a dictionary
dictionary = corpora.Dictionary(data['review_text_clean'])
print(dictionary.num_nnz)

# create a document term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data['review_text_clean'] ]
print(len(doc_term_matrix))

# LDA model
lda = gensim.models.ldamodel.LdaModel

# Fit model
num_topics = 7
ldamodel = lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50, minimum_probability=0)

#print(ldamodel.print_topics(num_topics=num_topics))

# lda visualization
lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False, mds='mmds')
pyLDAvis.save_html(lda_display, 'LDA_Visualization.html') # save a visualization to a standalone html file



"""
# Assigns the topics to the documents in corpus
lda_corpus = ldamodel[doc_term_matrix]

#print([doc for doc in lda_corpus])

scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))

threshold = sum(scores)/len(scores)
print(threshold)

cluster1 = [j for i,j in zip(lda_corpus,data.index) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,data.index) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,data.index) if i[2][1] > threshold]
cluster4 = [j for i,j in zip(lda_corpus,data.index) if i[3][1] > threshold]
cluster5 = [j for i,j in zip(lda_corpus,data.index) if i[4][1] > threshold]
cluster6 = [j for i,j in zip(lda_corpus,data.index) if i[5][1] > threshold]
cluster7 = [j for i,j in zip(lda_corpus,data.index) if i[6][1] > threshold]
"""