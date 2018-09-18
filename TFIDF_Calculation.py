import math


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist)/(1+ n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def sim(r1, r2):
    all_words = set(r1["words"]).union(set(r2["words"]))

    numerator = 0
    r1_denominator = 0
    r2_denominator = 0
    for word in all_words:
        if word in r1["words"] and word in r2["words"]:
            numerator += r1["words"][word] * r2["words"][word]

        if word in r1["words"]:
            r1_denominator += r1["words"][word] ** 2
        if word in r2["words"]:
            r2_denominator += r2["words"][word] ** 2

    denominator = (r1_denominator * r2_denominator)**0.5

    return numerator / denominator
