from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from FeatureVectorGeneration import calculateFeatureVector
from create_wordlist import loadWordlist
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw





def vectorDistance(t1, t2):
    sum = 0
    for i in range(len(t1)):
        sum += (t1[i] - t2[i]) ** 2
    return sqrt(sum)


def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (vectorDistance(s1[i],s2[j]))**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

wordlist = loadWordlist()
distances = list()
for i in wordlist[4:1000]:
    dist, x = fastdtw(np.array(i.featureVector), np.array(wordlist[3].featureVector), dist=euclidean)
    distances.append(dist)
    print(distances)
distances = sorted(distances)
print(distances[:20])
plt.plot(distances)
plt.show()
