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

We did a set of test runs to fine tune the parameters which we needed to get the similar images. Therefore we looked for the `keywords.txt` images inside the train set.

The results of the runs:
Percentage 0.1 Precision 0.2653061224489796 recall 0.014145810663764961 Accuracy 0.00013357994245787093
Percentage 0.2 Precision 0.2222222222222222 recall 0.015233949945593036 Accuracy 0.00014385532264693795
Percentage 0.3 Precision 0.2191780821917808 recall 0.017410228509249184 Accuracy 0.00016440608302507192 F1 0.0322580645161
Percentage 0.4 Precision 0.21978021978021978 recall 0.02176278563656148 Accuracy 0.0002055076037813399 F1 0.039603960396
Percentage 0.5 Precision 0.18487394957983194 recall 0.023939064200217627 Accuracy 0.0002260583641594739 F1 0.0423892100193
Percentage 0.6 Precision 0.15894039735099338 recall 0.026115342763873776 Accuracy 0.0002466091245376079 F1 0.0448598130841
Percentage 0.7 Precision 0.16201117318435754 recall 0.031556039173014146 Accuracy 0.0002979860254829429 F1 0.0528233151184
Percentage 0.8 Precision 0.16017316017316016 recall 0.04026115342763874 Accuracy 0.00038018906699547884 F1 0.064347826087
Percentage 0.9 Precision 0.1506849315068493 recall 0.04787812840043525 Accuracy 0.0004521167283189478
Percentage 1.0 Precision 0.12745098039215685 recall 0.056583242655059846 Accuracy 0.0005343197698314837

We decided to pick a percentage threshold of 0.9, because the precision is close to the one from lower percentages, but the recall is in a better spot.


TODO: Insert graphics, and stuff.

## Instructions
1. If you want to reproduce the experiment, just clone the git repo, and set the working directory to Exercise_3. If you want to reproduce the feature vector generation, just clear the content of the feature_vectors folder.
2. Adjust the `test` variable in `Evaluation.py` to the number of images you want to test.
3. Run Evaluation.py



## Results
Because we do not have the possibility to access performant computers / clusters. We had to limit the amount of test files we used for the results.
Therefore we picked 50 images at random from the test set.
TODO: Print results.


## Conclusion
Even with the optimal parameters, the performance was very bad.