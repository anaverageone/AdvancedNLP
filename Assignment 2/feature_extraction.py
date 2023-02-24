import pandas as pd
import numpy as np


def extract_features(file_path):
    '''Function for extracting the following features,
    - token's voice (based on dependency relation tags) & its position to predicate,
    - predicate lemma and its pos tag
    - pos tag of each token
    :param inputfile: string, file path

    :return a list of dictionaries
    '''
    #read in data to a pandas dataframe

    train_df = pd.read_csv(file_path, sep='\t', header=0, encoding='utf-8', quotechar='№')

    # change values in columns 'sent_id, Copy_ID, id' to integer
    train_df = train_df.astype({'Sent_ID':'int'})
    train_df = train_df.astype({'Copy_ID':'int'})
    train_df = train_df.astype({'ID':'int'})

    features_dict_list = []

    copy_id_list = train_df['Copy_ID'].unique()
    
    for num in copy_id_list:
        df_copy = train_df.loc[train_df['Copy_ID'] == num] # subset df for each sentence

        # get the length of sentence
        max_wds_count = df_copy['ID'].max()

        #find ID value of the predicate
        # pred_row = df_copy.loc[df_copy['UP:PRED']!= '_']
        # pred_id = pred_row['ID']
        pred_item = df_copy['ID'][df_copy['UP:PRED'].str.len() > 2]
        pred_id = None
        try:
            pred_id = pred_item.values[0]
        except Exception as err:
            print(' !!! WARNING !!! There is no predicate in sentence with copy_id:', num)
            continue

        print("pred_id:",pred_id)
        print("type pred_id:", type(pred_id))

        # create a new column "VOICE" and set all values to "0"
        df_copy['VOICE'] = '0' 

        # extract features within each sentence boundary
        for i in range(max_wds_count):
            features_dict = {}
            
            # row 'i'
            df_row = df_copy.iloc[i]

            # extract each token
            features_dict['token']=df_row['FORM']
            
            # extract POS, ALL TOKENS
            features_dict['pos'] = df_row['XPOS']

            # extract VOICE + POSITION TO PREDICATE
            if df_row['DEPREL'] == 'nsubj:pass':
                
                if i < (pred_id - 1):
                    df_row['VOICE'] = '1_before'
                elif i > (pred_id-1):
                    df_row['VOICE'] = '1_after'
                else:
                    df_row['VOICE'] = '1_same'
            else:
                if i < (pred_id-1):
                    df_row['VOICE'] = '0_before'
                elif i > (pred_id-1):
                    df_row['VOICE'] = '0_after'
                else:
                    df_row['VOICE'] = '0_same'

            features_dict['voice_position-to-pred'] = df_row['VOICE']

            # extract PREDICATE LEMMA + POS TAG
            if i == (pred_id - 1):
                features_dict['pred-lemma_pos'] = f"{df_row['LEMMA']}_{df_row['XPOS']}"
            
            features_dict_list.append(features_dict) 
        
    return features_dict_list

