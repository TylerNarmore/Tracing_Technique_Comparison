from os import listdir
from os.path import isfile, join
from textblob import TextBlob
from copy import deepcopy

from TFIDF_Calculation import *

from LDA_Modelling import *
from Corpus import Document, Corpus


def get_gannt_documents():
    """ Description:Get documents information from GANNT system
        Returns:    document_dictionary -  { Document ID :
                                                { "Name": Document Name,
                                                "Text": Document's Text }
                                            }
                    source_doc_names    -   List of names of the source documents
                    target_doc_names    -   List of names of the target documents
    """

    doc_id = 0
    source_file_directory = 'GANNT/high/'
    document_dictionary = {}

    source_doc_names = []
    # Gets a list of the source document file names
    source_file_names = [f for f in listdir(source_file_directory) if isfile(join(source_file_directory, f))]
    source_file_names.sort()
    # Iterates through the source document files and stores it's contents and names in document dictionary
    # Dictionary's key is an id number that increments with each document.
    for fileName in source_file_names:
        doc_name = fileName.rstrip('.txt')
        source_doc_names.append(doc_name)
        document_dictionary[doc_id] = {"Name": doc_name,
                                       "Text": open(source_file_directory+fileName, 'r').read().rstrip("\n")}
        doc_id += 1

    target_doc_names = []
    target_file_directory = 'GANNT/low/'
    # Gets a list of the target document file names
    target_file_names = [f for f in listdir(target_file_directory) if isfile(join(target_file_directory, f))]
    target_file_names.sort()
    # Iterates through the target document files and stores it's contents and names in document dictionary
    # Dictionary's key is an id number that increments with each document.


    for fileName in target_file_names:
        doc_name = fileName.rstrip('.txt')
        target_doc_names.append(doc_name)
        document_dictionary[doc_id] = {"Name": doc_name,
                                       "Text": open(target_file_directory + fileName, 'r').read().rstrip("\n")}
        doc_id += 1

    return document_dictionary, source_doc_names, target_doc_names


def find_links(threshold):
    """
    Calculates the TF-IDF score and finds the links above the threshold.

    :param threshold:
    :return valid_links:
    """
    gannt_doc_dict, source_doc_names, target_doc_names = get_gannt_documents()

    bloblist = []

    gannt_word_dict = {}

    # Creates teh text blob for each document
    for docID in range(len(gannt_doc_dict)):
        bloblist.append(TextBlob(gannt_doc_dict[docID]["Text"]))

    # Initializes the word dictionary that is used to calculate the TF_IDF scores
    for doc in gannt_doc_dict:
        gannt_word_dict[gannt_doc_dict[doc]["Name"]] = {"words": {}}

    # Calculates the word scores for each document and stores them in the gannt_word_dict
    # which will be used to calculate the TF_IDF scores
    for i, blob in enumerate(bloblist):
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        gannt_word_dict[gannt_doc_dict[i]["Name"]]["words"] = deepcopy(scores)
    valid_links = {}

    # Runs the TF_IDF calculation on each permutation of high->low level requirements
    for sourceDoc in source_doc_names:
        for targetDoc in target_doc_names:
            relation = sim(gannt_word_dict[sourceDoc], gannt_word_dict[targetDoc])

            # Adds links above the threshold to a dictionary using the tuple of source and target name's as the key
            if relation > threshold:
                valid_links[(sourceDoc, targetDoc)] = relation
    return valid_links


def create_gannt_corpus_obj():
    corpus = Corpus()
    doc_id = 0

    source_file_directory = 'GANNT/high/'
    target_file_directory = 'GANNT/low/'

    # Gets a list of the source document file names and sort by ID number
    source_file_names = [f for f in listdir(source_file_directory) if isfile(join(source_file_directory, f))]
    source_file_names.sort()

    # Iterates through the source document files and stores it's contents and names in document dictionary
    # Dictionary's key is an id number that increments with each document.
    for fileName in source_file_names:
        doc_name = fileName.rstrip('.txt')
        temp_document = Document(doc_id, doc_name, open(source_file_directory+fileName, 'r').read().rstrip("\n"))
        corpus.add_source_doc(temp_document)
        doc_id += 1

    # Gets a list of the target document file names and sort by ID number
    target_file_names = [f for f in listdir(target_file_directory) if isfile(join(target_file_directory, f))]
    target_file_names.sort()

    # Get and add to
    for fileName in target_file_names:
        doc_name = fileName.rstrip('.txt')
        temp_document = Document(doc_id, doc_name, open(target_file_directory+fileName, 'r').read().rstrip("\n"))
        corpus.add_target_document(temp_document)
        doc_id += 1

    return corpus


def main():

    threshold = 0.05

    threshold_links = {}

    # Location where csv files with the links are stored
    vsm_location = "GANNT_Answers/VSM/"
    corpus = create_gannt_corpus_obj()

    # Runs the TF_IDF calculations incrementing the threshold by 0.05 each iteration.
    # Stores the found links in a CSV file names at the threshold value * 100
    while threshold <= 1:
        file = open(vsm_location+str("{0:.2f}".format(threshold))[2:]+".csv", "w")

        valid_links = find_links(threshold)
        threshold_links[threshold] = valid_links
        print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for link in valid_links:
            source, target = link
            file.write(source + "," + target + "\n")

        threshold += 0.05
        file.close()

    lda_location = "GANNT_Answers/LDA/"
    # just testing LDA
    initialize()
    gannt_doc_dict, source_doc_names, target_doc_names = get_gannt_documents()
    for i in gannt_doc_dict:
        print(i, gannt_doc_dict[i])
    bow_corpus, dictionary = tokenize_lemmatize_docs(gannt_doc_dict)
    lda_scores = calculate_LDA_scores(bow_corpus, dictionary)
    threshold = 0.05
    while threshold <= 1:
        file = open(lda_location + str("{0:.2f}".format(threshold))[2:] + ".csv", "w")

        valid_links = find_links(threshold)
        threshold_links[threshold] = valid_links
        #print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for source_index in range(17):
            for target_index in range(17, 86):
                score = lda_scores[source_index, target_index]
                if score > threshold:
                    file.write(gannt_doc_dict[source_index]["Name"] + "," + gannt_doc_dict[target_index]["Name"] + "\n")

        threshold += 0.05
        file.close()


if __name__ == '__main__':
    main()
