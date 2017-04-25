# Exercise 3 - Keyword Spotting


## Packages


## Data
The data used in this exercise is provided in [this linked git repository](https://github.com/lunactic/PatRec17_KWS_Data). The three folders
- `ground-truth/`
- `images/`
- `task/`

have to be placed into the folder `Exercise_03/data/` on the local machine.

The folder `images/` contains 15 jpg-images, each represents a page from an ancient (handwritten) document. They are named with their page numbers `270.jpg`, `271.jpg`, ..., `304.jpg`.

The folder `ground-truth/` contains a file `transcription.txt` which contains the transcription of all words (on a character level) of the whole dataset. The format is as follows:

	- XXX-YY-ZZ: XXX = Document Number, YY = Line Number, ZZ = Word Number
	- Contains the character-wise transcription of the word (letters seperated with dashes)
	- Special characters denoted with s_
		- numbers (s_x)
		- punctuation (s_pt, s_cm, ...)
		- strong s (s_s)
		- hyphen (s_mi)
		- semicolon (s_sq)
		- apostrophe (s_qt)
		- colon (s_qo)

The folder `ground-truth/` also contains another folder, `locations/`, which contains 15 [svg-files](https://de.wikipedia.org/wiki/Scalable_Vector_Graphics), named with their page numbers as well `270.svg`, `271.svg`, ..., `304.svg`.
Each of those files contains the bounding boxes for all words on the according document page.

The folder `tasks/` contains three files, `keywords.txt`, `train.txt` and `valid.txt`. The files `train.txt` and `valid.txt` contain a splitting of the documents pages into a training and a validation set (by stating the page numbers).
The file `keywords.txt` contains a list of keywords, which each occurs at least once in both, the training and the validation dataset.

The content of the zipped file `cropped_words.zip` has to be placed into the folder `Exercise_03/data/` on the local machine.

## Description
In this exercise, a machine learning approach for spotting keywords in the provided documents will be developed. This approach will be tested on the provided training and validation dataset with the provided keywords that can be found for certain at least once in each set.

First, some preprocessing will be done. This includes binarization of the data and creation of word images. This is done with the sript `extract_images.py`.

Then a list containing all words can be created with the script `create_wordlist.py`.

## Instructions


## Results


## Conclusion
