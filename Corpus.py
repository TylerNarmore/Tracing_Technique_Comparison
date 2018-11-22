import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import nltk
from gensim import models
from gensim.matutils import cossim


class Corpus:
    def __init__(self, source_documents=[], target_documents=[]):
        np.random.seed(2018)

        nltk.download('wordnet')

        self.source_documents = source_documents
        self.target_documents = target_documents

        self.processed_documents = []
        self.processed_documents_dictionary = None
        self.bow_corpus = None
        self.corpus_lda = None
        self.lda_score_matrix = None

    def add_source_doc(self, document):
        self.source_documents.append(document)

    def add_target_document(self, document):
        self.target_documents.append(document)

    def generate_bowcorpus_and_dictionary(self):
        for document in self.source_documents:
            self.processed_documents.append(document.preprocessed_text)
        for document in self.target_documents:
            self.processed_documents.append(document.preprocessed_text)

        self.processed_documents_dictionary = gensim.corpora.Dictionary(self.processed_documents)
        self.processed_documents_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.bow_corpus = [self.processed_documents_dictionary.doc2bow(doc) for doc in self.processed_documents]

    def create_lda_score_matrix(self):
        log_ent = models.LogEntropyModel(self.bow_corpus)
        bow_corpus_ent = log_ent[self.bow_corpus]
        lda_model = gensim.models.LdaMulticore(bow_corpus_ent, num_topics=200, id2word=self.processed_documents_dictionary, passes=2, workers=2)
        self.corpus_lda = lda_model[log_ent[self.bow_corpus]]

        self.lda_score_matrix = np.zeros((len(self.bow_corpus), len(self.bow_corpus)))
        for i, par1 in enumerate(self.corpus_lda):
            for j, par2 in enumerate(self.corpus_lda):
                self.lda_score_matrix[i, j] = cossim(par1, par2)


class Document:
    def __init__(self, document_number, document_name, document_text):
        self.document_number = document_number
        self.document_text = document_text
        self.stemmer = SnowballStemmer("english")
        self.preprocessed_text = self.preprocess()

    def generate_textblob(self):
        # TODO generate textblob and return
        pass

    def lemmatize_stemming(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self):
        result = []
        for token in gensim.utils.simple_preprocess(self.document_text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result


