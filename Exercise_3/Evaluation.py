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

def worker(list):
    results = []
    for i in list:
        results.append((i, getSimilarImages(i, train)))
        print(threading.current_thread())
        print("finished one calculation")
    return results

start_time = timeit.default_timer()

wordlist = loadWordlist()
train, test = wordlistToDatasets(wordlist)
train = train[:400]
test = test[:8]

chunks = np.array_split(np.array(test),os.cpu_count())
chunks = [x.tolist() for x in chunks]

with Pool(2)as p:
    r = p.map(worker, chunks, 1)
    p.close()
    p.join()


results = []
for i in r:
    results.extend(i)

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

