import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
# this code is based on the codes used for Machine Learning for NLP course and for Applied Text Mining 1: Methods

def extract_features():
    '''
    
    '''
    return

def extract_AI_gold():
    '''
    
    '''
    return

def extract_AC_gold():
    '''
    
    '''
    return

def create_classifier():
    '''
    
    '''
    return

def classify_data():
    '''
    
    '''
    return

def save_the_output():
    '''
    
    '''
    return

def evaluation_report():
    '''
    
    '''
    return

def main(argv=None):
    '''
    
    '''
    # arguments 
    if argv == None:
        argv = sys.argv
    
    feature_extraction  = argv[1]
    train_AI = argv[2]
    train_AC = argv[3]

    # feature extraction 
    if feature_extraction:
        # extract features from the training set
        train_features = extract_features()
        # extract features from the test set
        test_features = extract_features()

    # ARGUMENT IDENTIFICATION
    if train_AI:
        # extract gold labels
        train_AI_labels = extract_AI_gold()
        test_AI_labels = extract_AI_gold()
        # train 1st LogReg
        AI_model, AI_vec = create_classifier()
        # test 1st LogReg
        classify_data()
        # save output of AI to new file
        save_the_output()
        # evaluate AI (need to have a column with all args having a uniform label bc binary)
        evaluation_report()

    # ARGUMENT CLASSIFICATION
    if train_AC:
        # extract gold labels
        train_AC_labels = extract_AC_gold()
        test_AC_labels = extract_AC_gold()
        # train 2nd LogReg
        AC_model, AC_vec = create_classifier() 
        # test 2nd LogReg
        classify_data()
        # save output of AC to a new file
        save_the_output()
        # evaluate AC
        evaluation_report()

if __name__ == '__main__':
    my_args = ['main.py', True, True, True]
    main(my_args)
