# Imports
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt
import sys

import pickle
from FeatureVectorGeneration import calculateFeatureVector

# paths
FEATURES_PATH = 'data/feature_vectors/'
WORD_PATH = 'data/cropped_words/'
TRANSCRIPT_PATH = 'data/input_documents/transcription.txt'

# Change the default colormap to gray
plt.rcParams['image.cmap'] = 'gray'


def loadFeatureVector(id):
    if os.path.exists(FEATURES_PATH + id + '.pkl'):
        file = open(FEATURES_PATH + id + '.pkl','rb')
        vector = pickle.load(file)
        file.close()
        return vector
    else:
        vector = calculateFeatureVector(WORD_PATH + id + '.png')
        file = open(FEATURES_PATH + id + '.pkl','wb')
        pickle.dump(vector, file)
        file.close()
        return vector


class Word:
    """ A class for the words.
    
    Parameters
    ----------
    id : str
        The string has to be of the format XXX-YY-ZZ where
            XXX = document number
             YY = line number
             ZZ = word number
        that describes the exact location of the word.
     transcript : str
        Human readable string containing the transcription of the word on a character basis.
     
     Attributes
     ----------
     docNr : int
        The number of the document the word is in, extracted from parameter 'id'.
     lineNr : int
        The number of the line in the document the word is in, extracted from parameter 'id'.
     wordNr : int
        The number of the word in the line it is in, extracted from parameter 'id'.
     img : ndarray
        2D array containing the pixel values of the word's image.
     transcript : str
        Human readable string containing the transcription of the word on a character basis.
     featureVector : list of vectors
        The list of feature vectors for the specific image
     
    """
    def __init__(self, id, transcript):
        self.docNr = int(id[0:3])
        self.lineNr = int(id[4:6])
        self.wordNr = int(id[7:9])
        self.img = plt.imread(WORD_PATH + id + '.png')
        self.transcript = transcript.replace("\n","")
        self.featureVector = loadFeatureVector(id)

# create a list of all words

def loadWordlist():
    wordlist = list()

    with open(TRANSCRIPT_PATH) as f:
        for line in f:
            id, transcript = str.split(line, " ")
            wordlist.append(Word(id, transcript))
    return wordlist


# divide them into training and validation set

def wordlistToDatasets(wordlist, split_file):
    with open(split_file) as v:
        valid_docNr = v.read().splitlines()

    del v

    valid_docNr = [int(x) for x in valid_docNr]

    train = list()
    valid = list()

    for word in wordlist:
        if word.docNr in valid_docNr:
            valid.append(word)
        else:
            train.append(word)

    with open('data/validation/keywords.txt') as t:
        keywords = t.read().splitlines()

        for word in valid:
            if word.transcript not in keywords:
                valid.remove(word)

    return train, valid
