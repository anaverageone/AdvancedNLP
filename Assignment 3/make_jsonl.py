import pandas as pd
from split_files import remove_comments,split_by_predicates
import sys
# this script uses code previously made for Assignment 2 of the Advanced NLP course

def bio_labels(label):
    '''
    This function adds BIO format to a gold label for SRL task.
    :param str label: gold label to modify

    :return str label: modified label
    '''
    if label != 'O':
        return f'B-{label}'
    else:
        return label
    
def write_jsonl(filepath, data_type):
    '''
    This function writes a new .jsonl file to feed sentences into a transformer model.
    :param str filepath: path to the .tsv file with sentences split by predicates
    :param str data_type: 'train' or 'test'

    :return: None 
    '''
    # read in the file as a dataframe
    df = pd.read_csv(filepath,sep='\t',encoding='utf-8',engine='python',quotechar='№',header=0)
    # split the data into sentences with separate predicate
    sentences = df.groupby(['Copy_ID'])
    with open(f'../data/{data_type}.jsonl', 'w',encoding='utf-8') as outfile:
        for name, sentence in sentences:
            # create a dictionary for each sentence (aka a line in the jsonl file)
            sent_dict = {}
            seq_words = sentence['FORM'].tolist()
            # add the list of tokens for the sentence
            sent_dict['seq_words'] = seq_words
            labels = sentence['UP:ARGHEADS'].tolist()
            # add BIO conventions to the gold labels
            bio = map(bio_labels,labels)
            # add the list of BIO-modified labels 
            sent_dict['BIO'] = list(bio)
            pred_row = sentence.loc[sentence['UP:PRED'] != '_']
            if pred_row.empty:
                # if no predicate sense
                pred = '_'
                pred_position = 'no'
            else:
                pred = pred_row['UP:PRED']
                pred = pred.values[0]
                pred_position = pred_row['ID'].astype('int32')
                pred_position = pred_position.values[0]
                # indexing in the list starts with 0 but in the original data it starts with 1
                pred_position = pred_position-1
            pred_sense = [pred_position,pred]
            # add predicate position in the 'seq_words' key and the predicate sense
            sent_dict['pred_sense'] = pred_sense
            outfile.write(f'{sent_dict}\n')
    

def main(argv=None):
    '''
    This function preprocesses the original datasets and creates .jsonl files to feed into transformer model for Assignment 3.

    :param list argv: list of arguments in the form ['make_jsonl.py', bool_1, bool_2]

    bool_1: whether to split original data by predicates
    bool_2: whether to create .jsonl files
    '''
    if argv == None:
        argv = sys.argv
    
    split_sentences = argv[1]
    create_jsonl = argv[2]

    if split_sentences:
        # paths to original datasets
        trainpath = '../data/en_ewt-up-train.conllu'
        testpath = '../data/en_ewt-up-test.conllu'
        # paths to datasets without comments
        train_no_comment = '../data/train_without_comments.tsv'
        test_no_comment = '../data/test_without_comments.tsv'
        # remove commented out lines from the original data
        remove_comments(trainpath,train_no_comment)
        remove_comments(testpath,test_no_comment)
        # split sentences in the datasets by predicates
        split_by_predicates(train_no_comment,'train')
        split_by_predicates(test_no_comment,'test')
    if create_jsonl:
        trainpath = '../data/train_split.tsv'
        testpath = '../data/test_split.tsv'
        write_jsonl(trainpath, 'train')
        write_jsonl(testpath, 'test')

if __name__ == '__main__':
    my_args = ['make_jsonl.py', False, True]
    main(my_args)
