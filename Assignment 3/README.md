# Advanced NLP, P4, 2023, Assignment 3 - BERT fine-tuned for SRL
Master of Text Mining, VU University
Task: Semantic Role labeling - BERT fine-tuning
gitHub Repository link: https://github.com/anaverageone/AdvancedNLP-Group3/tree/main/Assignment%203

### Group 3 members:
Meruyert Nurberdikhanova (2779728),
Cecilia Kuan (2770087),
Long Ma (2761790),
Siti Nurhalimah (2692449)

### Files
Zip file (also in repository link above) holds following files for BERT fine-tuning for SRL:
- ```split_files.py```: 
    - a script containing functions called in ```main.py``` to pre-process original training (en_ewt-up-train.conllu) & test dataset (en_ewt-up-test.conllu), script is an exact copy from assignment 2.
    - output - train_split.tsv and test_split.tsv (respectively) for feature extraction.  
- ```make_jsonl.py```: 
    - a sript preprocesses the original datasets and creates .jsonl files to feed into transformer model for Assignment 3.
    - output - .jsonl files for training and test datasets
- ```train.py```:
    - a script to fine-tune (train) a BERT-based model - preprocess the .jsonl datasets with BertTokenizer, fine-tune (train) the model 'BertForTokenClassification', evaluate the trained model's performance with the dev set, record progress in log files, and save the trained model file
- ```predict.py```:
    - a script to make predication on SRL task with the fine-tuned model from 'train.py' - preprocess the .jsonl test set with BertTokenizer, and provide prediction of the test set. Progress, predictions, and the evaluation of the prediction (in the value of precision, recall, and f1-score) are recorded in log files.

### Process from data processing, training the model, to making predictions



