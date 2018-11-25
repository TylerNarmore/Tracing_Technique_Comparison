import xml.etree.ElementTree as ET
import csv


def extract_gannt_actual_results_pairs(file):
    tree = ET.parse(file)
    root = tree.getroot()

    links = []
    for i in root.iter('link'):
        links.append((i[0].text, i[1].text))

    return links


def extract_icebreaker_actual_results_pairs(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        links =[]
        for source, target in csv_reader:
            links.append((source, target))
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
    if len(tool_pairs) != 0:
        precision = (len(true_positives)/(len(tool_pairs)))
    else:
        precision = -1
    recall = (len(true_positives)/(len(actual_pairs)))

    return precision, recall


gannt_actual = extract_gannt_actual_results_pairs("GANNT/answer_req_req.xml")
icebreaker_actual = extract_icebreaker_actual_results_pairs("IceBreaker/Requirements2ClassMatrix.csv")
results_file = open("results.csv", "w")
header = "System,Technique,Threshold,Recall,Precision\n"
results_file.write(header)

# Gannt Precision and recall
i = 0.00
while i < 1:
    file_name = ("%0.2f"%i)[2:]+".csv"
    tool = extract_tool_results_pairs("Gannt_Answers/VSM/" + file_name)
    precision, recall = get_precision_and_recall(gannt_actual, tool)
    if precision != -1:
        file_line = "GANNT,VSM,0." + str(file_name[:2]) + "," + "%0.03f" % recall + "," + "%0.03f" % precision + "\n"
        results_file.write(file_line)
    i += .05

i = 0.00
while i < 1:
    file_name = ("%0.2f"%i)[2:]+".csv"
    tool = extract_tool_results_pairs("Gannt_Answers/LDA/" + file_name)
    precision, recall = get_precision_and_recall(gannt_actual, tool)
    if precision != -1:
        file_line = "GANNT,LDA,0." + str(file_name[:2]) + "," + "%0.03f" % recall + "," + "%0.03f" % precision + "\n"
        results_file.write(file_line)
    i += .05

i = 0.00
while i < 1:
    file_name = ("%0.2f"%i)[2:]+".csv"
    tool = extract_tool_results_pairs("IceBreaker_Answers/VSM/" + file_name)
    precision, recall = get_precision_and_recall(icebreaker_actual, tool)
    if precision != -1:
        file_line = "IceBreaker,VSM,0." + str(file_name[:2]) + "," + "%0.03f" % recall + "," + "%0.03f" % precision + "\n"
        results_file.write(file_line)
    i += .05

i = 0.00
while i < 1:
    file_name = ("%0.2f"%i)[2:]+".csv"
    tool = extract_tool_results_pairs("IceBreaker_Answers/LDA/" + file_name)
    precision, recall = get_precision_and_recall(icebreaker_actual, tool)
    if precision != -1:
        file_line = "IceBreaker,LDA,0." + str(file_name[:2]) + "," + "%0.03f" % recall + "," + "%0.03f" % precision + "\n"
        results_file.write(file_line)
    i += .05