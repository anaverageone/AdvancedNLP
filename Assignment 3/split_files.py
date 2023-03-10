import pandas as pd
# this is just an exact copy of the script used in Assignment 2

def remove_comments(readpath, writepath):
    '''
    This function removes comment lines out of the original dataset and saves modified dataset into a new file.

    :param str readpath: path to the original dataset
    :param str writepath: path to the new file

    :return: None
    '''
    outfile = open(writepath,'w', encoding='utf-8')
    for line in open(readpath,'r',encoding='utf-8'):
        if line.startswith('#'):
            continue
        else:
            outfile.write(line)
    outfile.close()

def sent(row, list_firsts):
    '''
    This function creates sentence numbers to indicate sentence boundaries for the token-level dataset.

    :param Series row: row of the datasets
    :param list list_firsts: list of indices of first tokens in each sentence of the dataset

    :return int sent_num: sentence number in the dataset
    '''
    for ix, first in enumerate(list_firsts):
        if row.name == first:
            sent_num = ix+1
            return sent_num

def split_by_predicates(filepath, data_type):
    '''
    This function modifies the original dataset by splitting each sentence by its predicates and creating its copies, which is then saved into a new file.

    :param str filepath: path to the original dataset
    :param str data_type: 'train' or 'test'

    :return: None
    '''
    header = ['ID','FORM','LEMMA','UPOS','XPOS','FEATS','HEAD','DEPREL','DEPS','MISC','UP:PRED','UP:ARGHEADS_1','UP:ARGHEADS_2','UP:ARGHEADS_3','UP:ARGHEADS_4',
              'UP:ARGHEADS_5','UP:ARGHEADS_6', 'UP:ARGHEADS_7','UP:ARGHEADS_8','UP:ARGHEADS_9','UP:ARGHEADS_10','UP:ARGHEADS_11','UP:ARGHEADS_12','UP:ARGHEADS_13',
              'UP:ARGHEADS_14','UP:ARGHEADS_15','UP:ARGHEADS_16','UP:ARGHEADS_17','UP:ARGHEADS_18','UP:ARGHEADS_19','UP:ARGHEADS_20','UP:ARGHEADS_21','UP:ARGHEADS_22',
              'UP:ARGHEADS_23','UP:ARGHEADS_24','UP:ARGHEADS_25','UP:ARGHEADS_26','UP:ARGHEADS_27','UP:ARGHEADS_28','UP:ARGHEADS_29','UP:ARGHEADS_30','UP:ARGHEADS_31',
              'UP:ARGHEADS_32','UP:ARGHEADS_33','UP:ARGHEADS_34','UP:ARGHEADS_35']
    # header names taken from: 
    # https://universaldependencies.org/format.html
    # https://universalpropositions.github.io/
    df = pd.read_csv(filepath, sep='\t', names=header, encoding='utf-8',quotechar='№')
    # remove rows with CopyOf= in the MISC column
    df = df[~df.MISC.str.contains('CopyOf=',na=False)]
    # replace NaN values in predicate column with '_'
    df['UP:PRED'].fillna('_',inplace=True)
    # tokens that are at the beginning of each sentence
    start_of_sent = df.index[df['ID'] == 1 ].tolist()
    # get sentence IDs for each sentence
    df['Sent_ID'] = df.apply(lambda row: sent(row, start_of_sent), axis=1)
    # move the sentence number column to the front
    # adapted from: https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
    cols = list(df)
    cols.insert(0,cols.pop(cols.index('Sent_ID')))
    df = df.loc[:,cols]
    # fill NaN values for all tokens that are not at the beginning of the sentence
    df.Sent_ID.ffill(inplace=True)
    # group data by sentences
    sentences = df.groupby(['Sent_ID'])
    # the headers needed for split data except for UP:ARGHEADS that will be added later
    header_split_df = ['Sent_ID','ID','FORM','LEMMA','UPOS','XPOS','FEATS','HEAD','DEPREL','DEPS','MISC','UP:PRED']
    # create the new df to store splitted data in
    # start with the first sentence (based on the fact that we know it only has one predicate in it)
    first = sentences.get_group(1)
    new_df = first.filter(items=header_split_df)
    new_df['UP:ARGHEADS'] = first['UP:ARGHEADS_1']
    # iterate through grouped sentences
    for name, sentence in sentences:
        if name == 1: # skip the first sentence
            continue
        predicates_list = sentence['UP:PRED'].tolist()
        if '_' in predicates_list:
            predicates_list = [value for value in predicates_list if value != '_']
        if not predicates_list: # if empty bc there are no predicates in the sentence
            sentence_df = sentence.filter(items=header_split_df)
            sentence_df['UP:ARGHEADS'] = '_'
            new_df = pd.concat([new_df,sentence_df])
        else:
            predicates_dict = {k:v for k,v in enumerate(predicates_list)}
            for ix, p in predicates_dict.items():
                sentence_df = sentence.filter(items=header_split_df)
                replace_dict = dict(predicates_dict)
                del replace_dict[ix]
                sentence_df['UP:ARGHEADS'] = sentence[f'UP:ARGHEADS_{ix+1}']
                new_df = pd.concat([new_df,sentence_df])
    # remove the extra predicates with loc function
    new_df.loc[new_df['UP:ARGHEADS'] != 'V', 'UP:PRED'] = '_'
    # reset the index of the dataframe (bc it is copied for each sentence copy)
    new_df.reset_index(drop=True,inplace=True)
    # add copy id? aka just normal sentence id that will count the copies as sentences too
    # tokens that are at the beginning of each sentence
    start_of_sentence = new_df.index[new_df['ID'] == 1 ].tolist()
    # get sentence IDs for each sentence
    new_df['Copy_ID'] = new_df.apply(lambda row: sent(row, start_of_sentence), axis=1)
    # move the sentence number column to the front
    # adapted from: https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
    cols = list(new_df)
    cols.insert(0,cols.pop(cols.index('Copy_ID')))
    new_df = new_df.loc[:,cols]
    # fill NaN values for all tokens that are not at the beginning of the sentence
    new_df.Copy_ID.ffill(inplace=True)
    # replace _ with O for gold argument labels
    new_df.loc[new_df['UP:ARGHEADS'] == '_', 'UP:ARGHEADS'] = 'O'
    # save to a tsv file
    new_df.to_csv(f'../data/{data_type}_split.tsv',sep='\t',index=False,encoding='utf-8',quotechar='№')