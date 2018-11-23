import xml.etree.ElementTree as ET
import csv
from pprint import pprint


def extract_actual_results_pairs(file):
    tree = ET.parse(file)
    root = tree.getroot()

    links = []
    for i in root.iter('link'):
        links.append((i[0].text, i[1].text))

    return links


def extract_tool_results_pairs(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        links = []
        for row in csv_reader:
            links.append((row[0], row[1]))

        return links


def get_precision_and_recall(actual_pairs, tool_pairs):
    true_positives = list(set(actual_pairs)&set(tool_pairs))
    if (len(tool_pairs) != 0):
        precision = (len(true_positives)/(len(tool_pairs)))
    else:
        precision = -1
    recall = (len(true_positives)/(len(actual_pairs)))

    return precision, recall

actual = extract_actual_results_pairs("GANNT/answer_req_req.xml")
i = 0.05
while i < 1:
    file_name = ("%0.2f"%i)[2:]+".csv"
    tool = extract_tool_results_pairs("GANNT_Answers/LDA/" + file_name)
    precision, recall = get_precision_and_recall(actual, tool)
    print(file_name[:2], "\tPrecision\t%0.4f\t\t"%precision, "Recall\t\t%0.4f"%recall, sep="")
    i += .05

get_precision_and_recall(actual, tool)