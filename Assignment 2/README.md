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
- ```split_files.ipynb```: a Jupyter notebook to pre-process original training (en_ewt-up-train.conllu) & test dataset (en_ewt-up-test.conllu) to train_split.tsv and test_split.tsv respectively for feature extraction.  

- ```extract_gold_labels.ipynb```: 
- ```feature_extraction.py```: a python script to extract features on token-level. Output is used in ```create_classifier.ipynb``` for traiming the classifier.
- ```create_classifier.ipynb```: a Jupyter notebook to create classifier with logistic regression algorithm.  
- ```main.py```:

