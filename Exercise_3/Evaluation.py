import random
import timeit
from multiprocessing.pool import Pool

from SimilarImages import getSimilarImages
from create_wordlist import loadWordlist, wordlistToDatasets


RESULTS_PATH = 'results.txt'

"""
This Class is responsible for handling the evaluation. It calculates the tp, fp, fn, tn and
with it the accuracy.
It then prints everything into the desired form and stores it into a text file.
"""

def getConfusionMatrix(i, retrieved, train):
    tp = [x[1].transcript for x in retrieved].count(i.transcript)
    fp = len(retrieved) - tp
    fn = [x.transcript for x in train].count(i.transcript) - tp
    tn = len(train) - tp - fn - fp
    return tp, fp, tn, fn


def worker(i):
    return (i, getSimilarImages(i, train))


start_time = timeit.default_timer()

wordlist = loadWordlist()
train, test = wordlistToDatasets(wordlist, 'data/validation/splits/valid.txt')


def random_subset(list, size):
    out = []
    for i in range(size):
        out.append(list[random.randint(0, len(list)-1)])
    return out


test = random_subset(test, 50)
#test= test[480:520]
results = Pool(2).map(worker, test)


def save_results(results):
    file = open(RESULTS_PATH, "w+")
    for result in results:
        out = result[0].transcript
        for tuple_ in result[1]:
            out = out + ' ' + tuple_[1].transcript + ',' + str(tuple_[0])
        print(out, file=file)
    file.close()


save_results(results)

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

precision = tp / (tp + fp)
if tp + fn == 0:
    recall = None
else:
    recall = tp / (tp + fn)
accuracy = tp / (tp + fp + fn + tn)

print("Precision", precision, "recall", recall, "Accuracy", accuracy)
