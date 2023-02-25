import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from feature_extraction import extract_features
# this code is based on the codes used for Machine Learning for NLP course and for Applied Text Mining 1: Methods


def extract_AI_gold(filepath):
    '''
    This function extracts gold labels for Argument Identification task from a specified file path.
    The labels are 'O' and 'ARG' for binary classification task.

    :param str filepath: path to the file
    :return list target_list: list containing the gold labels
    '''
    file_df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№')
    argument_list = file_df['UP:ARGHEADS'].tolist()
    target_list = []
    for argument in argument_list:
        if argument != 'O':
            target = 'ARG'
            target_list.append(target)
        else:
            target_list.append(argument)
    
    return target_list

def extract_AC_gold(filepath):
    '''
    This function extracts gold labels for Argument Classification task from a specified file path.
    The labels are 'O', 'V', and labels of arguments from PropBank.

    :param str filepath: path to the file
    :return list target_list: list containing the gold labels
    '''
    file_df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№')
    target_list = file_df['UP:ARGHEADS'].tolist()

    return target_list

def create_classifier(train_features, train_targets):
    '''
    This function trains a Logistic Regression classifier on specified features and gold labels.

    :param list train_features: list of dictionaries of features
    :param list train_targets: list of gold labels

    :return model: fitted Logistic Regression model
    :return vec: fitted dictionary vectoriser
    '''
    vec = DictVectorizer()
    model = LogisticRegression()
    features_vectorised = vec.fit_transform(train_features)
    model.fit(features_vectorised, train_targets)

    return model, vec

def classify_data(model, vectoriser, test_features, testpath, outpath):
    '''
    This function tests a given Logistic Regression classifier on specified features and saves the predictions into a new file.

    :param model: fitted Logistic Regression model
    :param vectoriser: fitted dictionary vectoriser
    :param list test_features: list of dictionaries of features
    :param str testpath: path to the test set
    :param str outpath: path to the prediction file

    :return: None
    '''
    features_vectorised = vectoriser.transform(test_features)
    predictions = model.predict(features_vectorised)
    test_df = pd.read_csv(testpath, sep='\t', header=0, encoding='utf-8', quotechar='№')
    test_df['Predictions'] = predictions
    test_df.to_csv(outpath, sep='\t', index=False, encoding='utf-8', quotechar='№')

def evaluation_report(filepath, gold_labels, task_name):
    '''
    This function creates a classification report and a confusion matrix for specified predictions and gold labels.

    :param str filepath: path to the file with predictions
    :param list gold_labels: list of gold labels 
    :param str task_name: name to specify while printing

    :return: None
    '''
    df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№')
    predictions = df['Predictions'].tolist()
    report = classification_report(predictions, gold_labels)
    conf_matrix = confusion_matrix(predictions, gold_labels)
    print(f'------------- Classification report for {task_name} ------------------')
    print(report)
    print(f'------------- Confusion matrix for {task_name} ------------------')
    print(conf_matrix)

def main(argv=None):
    '''
    This function executes all the code needed for Semantic Role Labelling task.

    :param list argv: list of arguments in the form ['main.py', bool_1, bool_2, bool_3]

    bool_1: whether to execute feature extraction
    bool_2: whether to train & evaluate a classifier for Argument Identification
    bool_3: whether to train & evaluate a classifier for Argument Classification
    '''
    # arguments 
    if argv == None:
        argv = sys.argv
    
    feature_extraction  = argv[1]
    train_AI = argv[2]
    train_AC = argv[3]

    trainpath = '../data/train_split.tsv'
    testpath = '../data/test_split.tsv'
    # feature extraction 
    if feature_extraction:
        # extract features from the training set
        print('Extracting training features...')
        train_features = extract_features(trainpath)
        print('Done extracting training features!')
        # extract features from the test set
        print('Extracting test features...')
        test_features = extract_features(testpath)
        print('Done extracting test features!')

    # ARGUMENT IDENTIFICATION
    if train_AI:
        # extract gold labels
        print('Extracting AI gold labels...')
        train_AI_labels = extract_AI_gold(trainpath)
        test_AI_labels = extract_AI_gold(testpath)
        print('Done extracting AI gold labels!')
        # train 1st LogReg
        print('Training an AI classifier...')
        model_AI, vec_AI = create_classifier(train_features, train_AI_labels)
        print('Done training an AI classifier!')
        # test 1st LogReg and save the output
        path_AI_predictions = '../data/AI_predictions.tsv'
        print('Testing an AI classifier...')
        classify_data(model_AI,vec_AI,test_features,testpath,path_AI_predictions)
        print('Done testing an AI classifier!')
        # evaluate AI (need to have a column with all args having a uniform label bc binary)
        evaluation_report(path_AI_predictions,test_AI_labels, 'Argument Identification')

    # ARGUMENT CLASSIFICATION
    if train_AC:
        # extract gold labels
        print('Extracting AC gold labels...')
        train_AC_labels = extract_AC_gold(trainpath)
        test_AC_labels = extract_AC_gold(testpath)
        print('Done extracting AC gold labels!')
        # train 2nd LogReg
        print('Training an AC classifier...')
        model_AC, vec_AC = create_classifier(train_features, train_AC_labels) 
        print('Done training an AC classifier!')
        # test 2nd LogReg and save the output
        path_AC_predictions = '../data/AC_predictions.tsv'
        print('Testing an AC classifier...')
        classify_data(model_AC,vec_AC,test_features, testpath, path_AC_predictions)
        print('Done testing an AC classifier!')
        # evaluate AC
        evaluation_report(path_AC_predictions,test_AC_labels, 'Argument Classification')

if __name__ == '__main__':
    my_args = ['main.py', True, True, True]
    main(my_args)
