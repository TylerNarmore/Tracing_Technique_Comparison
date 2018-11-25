import gensim
import numpy as np
from gensim import models
from collections import defaultdict


class Corpus:
    def __init__(self, source_documents=None, target_documents=None):
        np.random.seed(2018)

        if source_documents is not None:
            self.source_documents = source_documents
        else:
            self.source_documents = []
        if target_documents is not None:
            self.target_documents = target_documents
        else:
            self.target_documents = []

        self.dictionary = None
        self.corpus = None

        # VSM attributes
        self.vsm_tokenized_texts = []
        self.tfidf = None
        self.vsm_corpus_tfidf = None

        # LDA attributes
        self.lda_processed_documents = []
        self.lda_documents_dictionary = None
        self.bow_corpus = None
        self.corpus_lda = None
        self.lda_score_matrix = None
        self.lda = None
        self.lda_corpus_lda = None

    def add_source_document(self, document):
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

        self.dictionary = gensim.corpora.Dictionary(self.vsm_tokenized_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.vsm_tokenized_texts]

    def run_vsm(self):
        self.tfidf = models.TfidfModel(self.corpus)
        self.vsm_corpus_tfidf = self.tfidf[self.corpus]

        index = gensim.similarities.MatrixSimilarity(self.vsm_corpus_tfidf)

        for source_doc in self.source_documents:
            vec_bow = self.dictionary.doc2bow(source_doc.document_text.lower().split())
            vec_tfidf = self.tfidf[vec_bow]
            source_doc.vsm_results = index[vec_tfidf]

    def run_lda(self):
        self.lda = models.LdaModel(self.corpus)
        self.lda_corpus_lda = self.lda[self.corpus]
        i = 0
        index = gensim.similarities.MatrixSimilarity(self.lda_corpus_lda)
        for source_doc in self.source_documents:
            vec_bow = self.dictionary.doc2bow(source_doc.document_text.lower().split())
            vec_lda = self.lda[vec_bow]
            source_doc.lda_results = index[vec_lda]


class Document:
    def __init__(self, document_number, document_name, document_text):
        self.document_number = document_number
        self.document_name = document_name
        self.document_text = document_text

        self.vsm_results = None
        self.lda_results = None
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        stoplist = set('for a of the and to in'.split())
        self.tokens = [word for word in self.document_text.lower().split() if word not in stoplist]
