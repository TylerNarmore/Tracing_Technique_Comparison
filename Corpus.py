import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import nltk
from gensim import models
from gensim.matutils import cossim
from collections import defaultdict


class Corpus:
    def __init__(self, source_documents=[], target_documents=[]):
        np.random.seed(2018)

        nltk.download('wordnet')

        self.source_documents = source_documents
        self.target_documents = target_documents

        # VSM attributes
        self.vsm_tokenized_texts = []
        self.vsm_dictionary = None
        self.vsm_corpus = None

        # LDA attributes
        self.lda_processed_documents = []
        self.lda_documents_dictionary = None
        self.bow_corpus = None
        self.corpus_lda = None
        self.lda_score_matrix = None

    def add_source_doc(self, document):
        self.source_documents.append(document)

    def add_target_document(self, document):
        self.target_documents.append(document)

    def vsm_generate_dict_and_corpus(self):
        self.vsm_tokenized_texts = []
        frequency = defaultdict(int)

        for doc in self.target_documents:
            self.vsm_tokenized_texts.append(doc.tokens)
            for token in doc.tokens:
                frequency[token] += 1

        self.vsm_tokenized_texts = [[token for token in text if frequency[token] > 1]
                                    for text in self.vsm_tokenized_texts]

        self.vsm_dictionary = gensim.corpora.Dictionary(self.vsm_tokenized_texts)
        self.vsm_corpus = [self.vsm_dictionary.doc2bow(text) for text in self.vsm_tokenized_texts]

    def run_vsm(self):
        self.tfidf = models.TfidfModel(self.vsm_corpus)
        self.vsm_corpus_tfidf = self.tfidf[self.vsm_corpus]

        index = gensim.similarities.MatrixSimilarity(self.vsm_corpus_tfidf)

        for source_doc in self.source_documents:
            vec_bow = self.vsm_dictionary.doc2bow(source_doc.document_text.lower().split())
            vec_tfidf = self.tfidf[vec_bow]
            source_doc.vsm_results = index[vec_tfidf]

    def run_lda(self):
        self.lda = models.LdaModel(self.vsm_corpus)
        self.lda_corpus_lda = self.lda[self.vsm_corpus]

        index = gensim.similarities.MatrixSimilarity(self.lda_corpus_lda)
        for source_doc in self.source_documents:
            vec_bow = self.vsm_dictionary.doc2bow(source_doc.document_text.lower().split())
            vec_lda= self.lda[vec_bow]
            source_doc.lda_results = index[vec_lda]

    def lda_generate_bowcorpus_and_dictionary(self):
        for document in self.source_documents:
            self.lda_processed_documents.append(document.preprocessed_text)
        for document in self.target_documents:
            self.lda_processed_documents.append(document.preprocessed_text)

        self.lda_documents_dictionary = gensim.corpora.Dictionary(self.lda_processed_documents)
        self.lda_documents_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.bow_corpus = [self.lda_documents_dictionary.doc2bow(doc) for doc in self.lda_processed_documents]

    def lda_create_score_matrix(self):
        log_ent = models.LogEntropyModel(self.bow_corpus)
        bow_corpus_ent = log_ent[self.bow_corpus]
        lda_model = gensim.models.LdaMulticore(bow_corpus_ent, num_topics=200, id2word=self.lda_documents_dictionary, passes=2, workers=2)
        self.corpus_lda = lda_model[log_ent[self.bow_corpus]]

        self.lda_score_matrix = np.zeros((len(self.bow_corpus), len(self.bow_corpus)))
        for i, par1 in enumerate(self.corpus_lda):
            for j, par2 in enumerate(self.corpus_lda):
                self.lda_score_matrix[i, j] = cossim(par1, par2)


class Document:
    def __init__(self, document_number, document_name, document_text):
        self.document_number = document_number
        self.document_name = document_name
        self.document_text = document_text
        self.stemmer = SnowballStemmer("english")
        self.preprocessed_text = self.preprocess()

        self.vsm_results = None
        self.lda_results = None
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        stoplist = set('for a of the and to in'.split())
        self.tokens = [word for word in self.document_text.lower().split() if word not in stoplist]

    def lemmatize_stemming(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self):
        result = []
        for token in gensim.utils.simple_preprocess(self.document_text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result


