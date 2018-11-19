import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models
from pprint import pprint

from gensim.matutils import cossim
from gensim import matutils

def initialize():
    np.random.seed(2018)

    nltk.download('wordnet')
    global stemmer
    stemmer = SnowballStemmer("english")


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def tokenize_lemmatize_docs(doc_dict):
    gannt_word_dict = {}
    processed_docs = []
    # Creates teh text blob for each document
    for docID in range(len(doc_dict)):
        processed_text = preprocess(doc_dict[docID]['Text'])
        processed_docs.append(processed_text)
        doc_dict[docID]['ProcessedText'] = processed_text

    dictionary = gensim.corpora.Dictionary(processed_docs)

    count = 0
    # for k, v in dictionary.iteritems():
    #     print(k, v)
    #     count += 1

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # for j in range(len(bow_corpus)):
    #     for i in range(len(bow_corpus[j])):
    #         print("Word {} (\"{}\") appears {} time.".format(bow_corpus[j][i][0],
    #                                                          dictionary[bow_corpus[j][i][0]],
    #                                                          bow_corpus[j][i][1]))
    return bow_corpus, dictionary


# This gets to the matrix with the results, not sure if they are accurate though
def calculate_LDA_scores(bow_corpus, dictionary):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # for doc in corpus_tfidf:
    #     pprint(doc)

    log_ent = models.LogEntropyModel(bow_corpus)
    bow_corpus_ent = log_ent[bow_corpus]

    # Running LDA using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus_ent, num_topics=200, id2word=dictionary, passes=2, workers=2)
    corpus_lda = lda_model[log_ent[bow_corpus]]

    res = np.zeros((len(bow_corpus), len(bow_corpus)))
    for i, par1 in enumerate(corpus_lda):
        for j, par2 in enumerate(corpus_lda):
            res[i, j] = cossim(par1, par2)

    return res
