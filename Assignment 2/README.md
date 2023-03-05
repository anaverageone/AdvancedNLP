# Advanced NLP, P4, 2023, Assignment 2
Master of Text Mining, VU University
Task: Semantic Role labeling

### Group members:
Meruyert Nurberdikhanova (2779728),
Cecilia Kuan (2770087),
Long Ma (2761790),
Siti Nurhalimah (2692449)

### Files
Current repository holds following files for feature extractions:
- ```split_files.py```: a script containing functions called in ```main.py``` to pre-process original training (en_ewt-up-train.conllu) & test dataset (en_ewt-up-test.conllu) to train_split.tsv and test_split.tsv respectively for feature extraction.  
- ```feature_extraction.py```: a python script to create and extract features on token-level. Output is used in ```main.py``` for training the classifiers.
- ```main.py```: a script running all the core parts of our experiment (pre-processing, feature engineering, argument identification, argument classification, and overall results)
- ```output_main.txt```: a .txt file containing classification results from 2 steps and overall taken from the terminal output of running ```main.py``` with arguments: ['main.py', False, False, True, True, True]
- ```overall_confusion_matrix.tsv```: a .tsv file containing the overall experiment's confusion matrix, as it is too large to print out

