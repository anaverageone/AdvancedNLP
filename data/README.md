In this directory you can find the training and testing data we have used to complete the assignments.

Specifically, there are:
```en_ewt-up-train.conllu``` and ```en_ewt-up-test.conllu``` - the original datasets taken from https://github.com/UniversalPropositions/UP-1.0/tree/master/UP_English-EWT
```train_without_comments.tsv``` and ```test_without_comments.tsv``` - modified datasets without the comment lines that were indicated with '#'
```train_split.tsv``` and ```test_split.tsv``` - modified datasets with sentences split and copied by the amount of predicates present in each (these datasets are used as input for training and testing the models)