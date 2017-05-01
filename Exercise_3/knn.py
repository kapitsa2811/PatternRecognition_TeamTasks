from math import sqrt

from Exercise_3.FeatureVectorGeneration import calculateFeatureVector


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


#img1 = calculateFeatureVector("test.jpg")
#img2= calculateFeatureVector("test2.jpg")

#print(DTWDistance(img1,img2))