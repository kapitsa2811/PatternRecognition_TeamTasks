# Imports
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt
import sys

import pickle

sys.path.append('./Exercise_3')
from FeatureVectorGeneration import calculateFeatureVector

# Change the default colormap to gray
plt.rcParams['image.cmap'] = 'gray'


def loadFeatureVector(id):
    if os.path.exists('./Exercise_3/data/feature_vectors/' + id + '.pkl'):
        file = open('./Exercise_3/data/feature_vectors/' + id + '.pkl','rb')
        vector = pickle.load(file)
        file.close()
        return vector
    else:
        vector = calculateFeatureVector('./Exercise_3/data/cropped_words/' + id + '.png')
        file = open('./Exercise_3/data/feature_vectors/' + id + '.pkl','wb')
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
     featureVector : list ??? TODO
        ??? TODO
     
    """
    def __init__(self, id, transcript):
        self.docNr = int(id[0:3])
        self.lineNr = int(id[4:6])
        self.wordNr = int(id[7:9])
        self.img = plt.imread('./Exercise_3/data/cropped_words/' + id + '.png')
        self.transcript = transcript
        self.featureVector = loadFeatureVector(id)

# create a list of all words

wordlist = list()

with open('./Exercise_3/data/ground-truth/transcription.txt') as f:
    for line in f:
        id, transcript = str.split(line, " ")
        wordlist.append(Word(id, transcript))

del f, id, line, transcript

# divide them into training and validation set

with open('./Exercise_3/data/task/valid.txt') as v:
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

del word, wordlist, valid_docNr

