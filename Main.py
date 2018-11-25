from os import listdir
from os.path import isfile, join

from Corpus import Document, Corpus
import csv


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
        corpus.add_source_document(temp_document)
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


def create_icebreaker_corpus_obj():
    source_file = 'IceBreaker/Requirements.csv'
    target_file = 'IceBreaker/ClassDiagram.csv'
    corpus = Corpus()
    index = 0
    with open(source_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for name, desc in csv_reader:
            doc = Document(index, name, desc)
            corpus.add_source_document(doc)
        csv_file.close()

    with open(target_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for name, desc in csv_reader:
            doc = Document(index, name, desc)
            corpus.add_target_document(doc)
        csv_file.close()

    return corpus


def run_gannt_calculations():
    # Location where csv files with the links are stored
    vsm_location = "GANNT_Answers/VSM/"
    lda_location = "GANNT_Answers/LDA/"
    gannt_corpus = create_gannt_corpus_obj()
    gannt_corpus.vsm_generate_dict_and_corpus()
    gannt_corpus.run_vsm()
    gannt_corpus.run_lda()

    # # Runs the TF_IDF calculations incrementing the threshold by 0.05 each iteration.
    # # Stores the found links in a CSV file names at the threshold value * 100
    threshold = 0.00
    while threshold <= 1:
        file = open(vsm_location + str("{0:.2f}".format(threshold))[2:] + ".csv", "w")

        # print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for source in gannt_corpus.source_documents:
            index = 0
            for score in source.vsm_results:
                if score >= threshold:
                    file.write(source.document_name + "," + gannt_corpus.target_documents[index].document_name + "\n")
                index += 1

        threshold += 0.05
        file.close()

    threshold = 0.0
    while threshold <= 1:
        file = open(lda_location + str("{0:.2f}".format(threshold))[2:] + ".csv", "w")

        # print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for source in gannt_corpus.source_documents:
            index = 0
            for score in source.lda_results:
                if score >= threshold:
                    file.write(source.document_name + "," + gannt_corpus.target_documents[index].document_name + "\n")
                index += 1

        threshold += 0.05
        file.close()


def run_icebreaker_calculations():
    vsm_location = "IceBreaker_Answers/VSM/"
    lda_location = "IceBreaker_Answers/LDA/"

    icebreaker_corpus = create_icebreaker_corpus_obj()
    icebreaker_corpus.vsm_generate_dict_and_corpus()
    icebreaker_corpus.run_vsm()
    icebreaker_corpus.run_lda()

    threshold = 0.00
    while threshold <= 1:
        file = open(vsm_location + str("{0:.2f}".format(threshold))[2:] + ".csv", "w")

        print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for source in icebreaker_corpus.source_documents:
            index = 0
            for score in source.vsm_results:
                if score >= threshold:
                    file.write(source.document_name + "," +
                               icebreaker_corpus.target_documents[index].document_name + "\n")
                index += 1

        threshold += 0.05
        file.close()

    threshold = 0.0
    while threshold <= 1:
        file = open(lda_location + str("{0:.2f}".format(threshold))[2:] + ".csv", "w")

        print("******* Threshold = ", "{0:.2f}".format(threshold), " *******", sep='')
        for source in icebreaker_corpus.source_documents:
            index = 0
            for score in source.lda_results:
                if score >= threshold:
                    file.write(source.document_name + "," +
                               icebreaker_corpus.target_documents[index].document_name + "\n")
                index += 1

        threshold += 0.05
        file.close()


def main():
    run_gannt_calculations()
    run_icebreaker_calculations()


if __name__ == '__main__':
    main()
