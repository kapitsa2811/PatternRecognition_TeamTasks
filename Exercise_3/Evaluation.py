import pickle

from SimilarImages import calculateDistances, getThreshold, getSimilarImages
from create_wordlist import loadWordlist, wordlistToDatasets
from fastdtw import fastdtw
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

wordlist = loadWordlist()
train, test = wordlistToDatasets(wordlist)


def getConfusionMatrix(i, retrieved, train):
    tp = [x.transcript for x in retrieved].count(i.transcript)
    fp = len(retrieved)-tp
    fn = [x.transcript for x in train].count(i.transcript)- tp
    tn = len(train)- tp -fn
    return tp, fp, tn, fn

tp = 0
fp = 0
tn = 0
fn = 0

for i in test[:10]:
    retrieved = getSimilarImages(i, train[:1000])
    tpt, fpt, tnt, fnt = getConfusionMatrix(i, retrieved, train[:1000])
    tp += tpt
    fp += fpt
    tn += tnt
    fn += fnt

precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = tp / (tp + fp + fn + tn)

print("Precision" , precision , "recall", recall, "Accuracy" , accuracy)