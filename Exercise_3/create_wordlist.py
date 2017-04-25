# Imports
import numpy as np
import matplotlib.pyplot as plt

# Change the default colormap to gray
plt.rcParams['image.cmap'] = 'gray'


# A class for the words.
#
# @ id : string of the format XXX-YY-ZZ where
#          XXX = document number
#           YY = line number
#           ZZ = word number
#
# @ transcript : string containing the transcription of the word on a character basis

class Word:
    def __init__(self, id, transcript):
        self.docNr = int(id[0:3])
        self.lineNr = int(id[4:6])
        self.wordNr = int(id[7:9])
        self.img = plt.imread('./Exercise_3/data/cropped_words/'+id+'.jpg')
        self.transcript = transcript

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

del wordlist, word, valid_docNr
