import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from feature_extraction import extract_features
pd.options.mode.chained_assignment = None  # default='warn'
# this code is based on the codes used for Machine Learning for NLP course and for Applied Text Mining 1: Methods


def extract_AI_gold(filepath):
    '''
    This function extracts gold labels for Argument Identification task from a specified file path.
    The labels are 'O' and 'ARG' for binary classification task.

    :param str filepath: path to the file
    :return list target_list: list containing the gold labels
    '''
    file_df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№',engine='python')
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
    file_df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')
    file_df = file_df.loc[file_df['Predictions_AI'] == 'ARG']
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
    model = LogisticRegression(max_iter=500)
    features_vectorised = vec.fit_transform(train_features)
    model.fit(features_vectorised, train_targets)

    return model, vec

def classify_data(model, vectoriser, test_features, testpath, outpath, task_name):
    '''
    This function tests a given Logistic Regression classifier on specified features and saves the predictions into a new file.

    :param model: fitted Logistic Regression model
    :param vectoriser: fitted dictionary vectoriser
    :param list test_features: list of dictionaries of features
    :param str testpath: path to the test set
    :param str outpath: path to the prediction file
    :param str task_name: 'AI' for argument identification, 'AC' for argument classification

    :return: None
    '''
    features_vectorised = vectoriser.transform(test_features)
    predictions = model.predict(features_vectorised)
    test_df = pd.read_csv(testpath, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')
    if task_name == 'AI':
        test_df['Predictions_AI'] = predictions
    elif task_name == 'AC':
        # assign default O for all
        test_df['Predictions_AC'] = 'O'
        # get a subset of candidate arguments
        pred_df = test_df.loc[test_df['Predictions_AI'] == 'ARG']
        pred_df['Predictions_AC'] = predictions
        # add predictions back to the dataframe
        test_df.update(pred_df)
    else:
        raise ValueError('Incorrect task name!')
    test_df.to_csv(outpath, sep='\t', index=False, encoding='utf-8', quotechar='№')

def evaluation_report(filepath, gold_labels, task_name):
    '''
    This function creates a classification report and a confusion matrix for specified predictions and gold labels.

    :param str filepath: path to the file with predictions
    :param list gold_labels: list of gold labels 
    :param str task_name: name to specify while printing

    :return: None
    '''
    df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')
    if task_name == 'Argument Identification':
        predictions = df['Predictions_AI'].tolist()
    elif task_name == 'Argument Classification':
        predictions = df['Predictions_AC'].tolist()
    else:
        raise ValueError('Incorrect task name!')
    report = classification_report(predictions, gold_labels)
    conf_matrix = confusion_matrix(predictions, gold_labels)
    print(f'------------- Classification report for {task_name} ------------------')
    print(report)
    print(f'------------- Confusion matrix for {task_name} ------------------')
    print(conf_matrix)

def get_candidate_features(list_features, filepath):
    '''

    '''
    df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')
    candidate_indeces = df[df['UP:ARGHEADS'] != 'O'].index.tolist()
    candidate_list = []
    for item in list_features:
        if list_features.index(item) in candidate_indeces:
            # make a new list of dictionaries with features for only candidates
            candidate_list.append(item)
    return candidate_list
def main(argv=None):
    '''
    This function executes all the code needed for Semantic Role Labelling task.

    :param list argv: list of arguments in the form ['main.py', bool_1, bool_2]

    bool_1: whether to train & evaluate a classifier for Argument Identification
    bool_2: whether to train & evaluate a classifier for Argument Classification
    '''
    # arguments 
    if argv == None:
        argv = sys.argv
    
    train_AI = argv[1]
    train_AC = argv[2]

    # paths to datasets split by predicates
    trainpath = '../data/train_split.tsv'
    testpath = '../data/test_split.tsv'
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
        classify_data(model_AI, vec_AI, test_features, testpath, path_AI_predictions, 'AI')
        print('Done testing an AI classifier!')
        # evaluate AI (need to have a column with all args having a uniform label bc binary)
        evaluation_report(path_AI_predictions,test_AI_labels, 'Argument Identification')

    # ARGUMENT CLASSIFICATION
    if train_AC:
        # paths to train dataset and test dataset (with Argument Identification predictions)
        trainpath = '../data/train_split.tsv'
        testpath = '../data/AI_predictions.tsv'
        # extract AC features
        print('Extracting training features...')
        train_features = get_candidate_features(train_features, trainpath)
        print('Done extracting training features!')
        # extract features from the test set
        print('Extracting test features...')
        test_features = get_candidate_features(test_features, testpath)
        print('Done extracting test features!')
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
        classify_data(model_AC, vec_AC, test_features, testpath, path_AC_predictions, 'AC')
        print('Done testing an AC classifier!')
        # evaluate AC
        evaluation_report(path_AC_predictions,test_AC_labels, 'Argument Classification')

if __name__ == '__main__':
    my_args = ['main.py', True, True]
    main(my_args)
