# Exercise 3 - Keyword Spotting


## Packages
- numpy 1.12.0
- matplotlib 2.0.0
- scipy 0.19.0
- scikit-image 0.13.0
- re
- PIL
- os
- sys
- math

## Data
The data used in this exercise is provided in [this linked git repository](https://github.com/lunactic/PatRec17_KWS_Data). The three folders
- `ground-truth/`
- `images/`
- `task/`

have been rearranged so that the data so the project can be downloaded and is ready to go.


have to be placed into the folder `Exercise_03/data/` on the local machine.

The folder `input_documents/images/` contains 15 jpg-images, each represents a page from an ancient (handwritten) document. They are named with their page numbers `270.jpg`, `271.jpg`, ..., `304.jpg`.

The folder `input_documents/` contains a file `transcription.txt` which contains the transcription of all words (on a character level) of the whole dataset. The format is as follows:

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

The folder `input_documents/` also contains another folder, `locations/`, which contains 15 [svg-files](https://de.wikipedia.org/wiki/Scalable_Vector_Graphics), named with their page numbers as well `270.svg`, `271.svg`, ..., `304.svg`.
Each of those files contains the bounding boxes for all words on the according document page.

The folder `validation/` contains the file, `keywords.txt`. The file `keywords.txt` contains a list of keywords, which each occurs at least once in both, the training and the validation dataset.
The subfolder  `validation/splits/`, contains the files `train.txt` and `valid.txt`. The files `train.txt` and `valid.txt` contain a splitting of the documents pages into a training and a validation set (by stating the page numbers).

The folder `cropped_words/`  contains the pre processed words which are created in the extract_images.py.

The folder `feature_vectors/` contains the pre-created feature vectors for each image. The feature vectors are created using the sliding window approach. If you want to create the features for each image anew, just delete the content of the `feature_vectors/` folder.

The folder `temp_images/` contains the processed documents. The current set of documents was created using the sauvola method.

## Description
In this exercise, a machine learning approach for spotting keywords in the provided documents is developed. This approach is tested on the provided training and validation dataset with the provided keywords that can be found at least once in each set for sure.

First, some preprocessing is done. This includes binarization of the data and creation of word images. This is done with the sript `extract_images.py`. The output of this script is already provided in the the zip-file `Exercise_3/cropped_words.zip`. Several binarization approaches
have been explored. Besides a trivial quantile cutoff approach, the more sophisticated methods [Otsu](https://en.wikipedia.org/wiki/Otsu's_method) and [Niblack and Sauvola](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html) have been used. Comparing
the different results, Sauvolas' method is the best. The words contained in the `cropped_words` folder are binarized with this thresholding approach.

Then, the training and validation sets (each is a list of objects of type `Word`) are created with the script `create_wordlist.py`. The division into training and validation sets is according to the files `train.txt` and `valid.txt` from the folder `Exercise_3/task/`. An object of class `Word` contains the following attributes:

- `docNr` : string, the number of the document
- `lineNr` : the number of the line in the document
- `wordNr` : the number of the word in the line
- `img` : the image of the word as numpy array
- `transcript` : the transcription of the word on a character basis as described above
- `featureVector` : the created feature vectors of the word, using sliding window approach


The preprocessed images and the cropped words are then used and analyzed by calculating the featureVectors using the sliding window approach. Done in the FeatureVectorGeneration.py script.
To determine the similarity we used dynamic time warping done in SimilarImages.py with the help of fastdtw.py.

We did a set of test runs to fine tune the parameters which we needed to get the similar images.

TODO: Insert graphics, and stuff.

## Instructions
1. If you want to reproduce the experiment, just clone the git repo, and set the working directory to Exercise_3. If you want to reproduce the feature vector generation, just clear the content of the feature_vectors folder.
2. Adjust the `test` variable in `Evaluation.py` to the number of images you want to test.
3. Run Evaluation.py



## Results
TODO: Print results.


## Conclusion
Even with the optimal parameters, the performance was very bad.