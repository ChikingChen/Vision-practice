import numpy as np
import math

def calidf(bof, cluster):
    #  计算 idf 值
    idf = np.zeros((cluster)).tolist()
    for i in range(len(bof)):
        for j in range(len(bof[i])):
            if bof[i][j] != 0:
                idf[j] += 1
    for i in range(cluster):
        idf[i] = math.log((cluster + 1) / (idf[i] + 1)) + 1
    return idf

def caltfidf(bof, cluster, idf):
    #  计算 tf 值
    tf = np.zeros((len(bof), cluster)).tolist()
    for i in range(len(bof)):
        words = sum(bof[i])
        for j in range(len(bof[i])):
            tf[i][j] = bof[i][j] / words

    #  计算 tfidf 值
    tfidf = np.zeros((len(bof), cluster)).tolist()
    for i in range(len(bof)):
        for j in range(len(bof[i])):
            tfidf[i][j] = tf[i][j] * idf[j]

    return tfidf

def calsingletfidf(vector, cluster, idf):
    tf = np.zeros((cluster)).tolist()
    words = sum(vector)
    for i in range(len(vector)):
        tf[i] = vector[i] / words
    tfidf = np.zeros((cluster)).tolist()
    for i in range(len(vector)):
        tfidf[i] = tf[i] * idf[i]
    return tfidf