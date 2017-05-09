import os
import threading
import timeit
from multiprocessing.pool import Pool

import numpy as np
from multiprocessing import Process

from SimilarImages import getSimilarImages
from create_wordlist import loadWordlist, wordlistToDatasets


def getConfusionMatrix(i, retrieved, train):
    tp = [x.transcript for x in retrieved].count(i.transcript)
    fp = len(retrieved)-tp
    fn = [x.transcript for x in train].count(i.transcript)- tp
    tn = len(train)- tp -fn -fp
    return tp, fp, tn, fn

def worker(i):
    return (i, getSimilarImages(i, train))

start_time = timeit.default_timer()

wordlist = loadWordlist()
train, test = wordlistToDatasets(wordlist)
train = train
test = test[5:6]

results = Pool(2).map(worker, test)

tp = 0
fp = 0
tn = 0
fn = 0

for i in results:
    tpt, fpt, tnt, fnt = getConfusionMatrix(i[0], i[1], train)
    tp += tpt
    fp += fpt
    tn += tnt
    fn += fnt

elapsed = timeit.default_timer() - start_time

print(elapsed)

precision = tp/(tp+fp)
if tp + fn == 0:
    recall = None
else:
    recall = tp/(tp+fn)
accuracy = tp / (tp + fp + fn + tn)

print("Precision" , precision , "recall", recall, "Accuracy" , accuracy)

