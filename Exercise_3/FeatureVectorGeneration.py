import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




def calculateFeatureVector(filename):
    img= loadAndResizeImg(filename)
    return getFeatureVector(img)


def loadAndResizeImg(filename):
    img = np.asarray(Image.open(filename).resize((200, 200)))
    G = np.zeros((200, 200))

    # Where we set the RGB for each pixel
    G[img > 150] = True
    G[img <= 150] = False
    img = G
    return img


def getLowerContur(column):
    for i in range(len(column)):
        if column[i] == False:
            return i
        else:
            None


def getUpperContur(column):
    for i in reversed(range(len(column))):
        if column[i] == False:
            return i
        else:
            None


def getBWTransitions(column):
    counter = 0
    for i in range(len(column)-1):
        if column[i] != column[i+1]:
            counter = counter +1

    return counter


def getFractionOfBlackPxInWindow(column):
    counter = 0
    for i in range(len(column)):
        if column[i] == False:
            counter = counter +1

    return counter/len(column)


def getFractionOfBlackPxBtwLcAndUc(column, lc, uc):
    if lc == None or uc == None:
        return None
    counter = 0
    upperBound = uc + 1 if uc +1 <=200 else 200
    for i in range(lc,upperBound):
        if column[i] == False:
            counter = counter +1

    return counter/len(range(lc,upperBound))


def getGradientDifferenceLcUc(column):
    pass


def calculateFeatures(column):
    features = []
    #Lower contur
    features.append(getLowerContur(column))
    features.append(getUpperContur(column))
    features.append(getBWTransitions(column))
    features.append(getFractionOfBlackPxInWindow(column))
    features.append(getFractionOfBlackPxBtwLcAndUc(column, features[0], features[1]))
    return features
   # features.append(getGradientDifferenceLcUc(column))




def getFeatureVector(a):
    featureList = list()
    for i in range(200):
        featureList.append(calculateFeatures(a[:,i]))

    return featureList


