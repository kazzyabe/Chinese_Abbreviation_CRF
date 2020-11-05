# Chinese Abbreviation

## files
* prep.ipynb : jupyter notebook for preparing data for training nltk HMM tagger
* NLTK_HMM.ipynb : jupyter notebook for training and testing the NLTK HMM tagger for Abbreviation

## Data
* used https://github.com/zhangyics/Chinese-abbreviation-dataset.git 
* preprocessed dev, tr, te data are included in this repo
1. Cleaned_AbbOri_ : a cleaned abbreviation and the original word pairs
2. Tagged_ : tagged data, each characters are tagged with 'A' if included in abbreviation and 'N' otherwise
* results are also included
1. res_ : result files with abbreviation, original, gussed abbreviation

## USAGE
$ python abbreviate.py -f words.txt
* include words which you want to abbreviate in each line