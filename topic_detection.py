import pandas as pd
import numpy as np
import sklearn
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag
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
new_stopwords = ['athens', 'acropolis', 'u', 'acropoli','dont','its', 'history', 'acropolis', 'Athens', 'athena','place','see','that','day','lot','time','...','akropolis']
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

#keep only nouns and adj
def nouns_adj(text):
    # pull out only nouns and adjectives
    is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    all_nouns = [word for (word, pos) in pos_tag(text) if is_noun(pos)]
    return ' '.join(all_nouns)

data['review_text_clean']=data['review_text_clean'].apply(nouns_adj)

data['review_text_clean'] = [d.split() for d in data['review_text_clean']]


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
pyLDAvis.save_html(lda_display, 'Overall_Period_LDA_Visualization.html') # save a visualization to a standalone html file


