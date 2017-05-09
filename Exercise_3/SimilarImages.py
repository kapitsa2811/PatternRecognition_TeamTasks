from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from FeatureVectorGeneration import calculateFeatureVector
from create_wordlist import loadWordlist, wordlistToDatasets
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

percentage = 0.7
limit = 0.2

def calculateDistances(word, words):
    distances = list()
    for i in words:
        dist, x = fastdtw(np.array(word.featureVector), np.array(i.featureVector), dist=euclidean)
        distances.append((dist, i))
    return distances

def getThreshold(distances):
    distances.sort();
    largest = 0
    counter = 0
    for i in range(1, len(distances)):
        if (limit > abs(distances[i][0] - distances[i - 1][0])):
            counter += 1
            if counter > 4:
                largest = distances[i][0]
                break
        else:
            counter = 0
    if largest == 0:
        largest = 40
    return (abs(largest - distances[0][0])*percentage)+distances[0][0]

def showPlot(distances, threshhold):
    blah = [x[0] for x in distances]
    plt.plot(blah)
    plt.axhline(y=threshhold, color='r')
    plt.show()

def getMostSimilar(distances):
    threshhold =  getThreshold(distances)
    mostSimilar = list()
    distances.sort()
    #showPlot(distances, threshhold)
    for i in distances:
        if i[0] < threshhold:
            mostSimilar.append(i[1])
        else:
            break
    return mostSimilar


def getSimilarImages(word, words):
    distances = calculateDistances(word, words)
    return getMostSimilar(distances)