In this directory you can find the training and testing data we have used to complete the assignments.

Specifically, there are:
- ```en_ewt-up-train.conllu``` and ```en_ewt-up-test.conllu```: riginal datasets taken from https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT
- ```train_without_comments.tsv``` and ```test_without_comments.tsv```: modified datasets without the comment lines that were indicated with '#' (needed to split databy predicates)
- ```train_split.tsv``` and ```test_split.tsv```: modified datasets with sentences split and copied by the amount of predicates present in each (used to created new features)
- ```train_with_features.tsv``` and ```test_with_features.tsv```: modified datasets with additional features
- ```AI_predictions.tsv``` and ```AC_predictions.tsv```: test datasets with predictions saved for Argument Identification step and Argument Classification step
- ```model_inputs.txt``` and ```model_outputs.txt```: token sequences and predicted SRL labels of the fine-tuned BERT; this output was made on Google Colab