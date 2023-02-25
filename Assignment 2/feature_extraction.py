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

        # predicate for feature "PREDICATE & HEAD OF TOKEN"
        if not pred_item.empty:
            pred_id = pred_item.values[0]
            pred_form = df_copy.loc[df_copy['ID'] == pred_id,'FORM'].values[0]
        else:
            pred_form = 0

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

            
            ### #extract HEAD WD OF TOKEN + POS TAG ###
            head = None
            try:
                lemma = df_row ['LEMMA']
                head = int(df_row['HEAD'])
                xpos = df_row ['XPOS']
                token = df_row['FORM']
            except Exception as err:
                print("==== BLAH BLAH ====")
                print(err)
                raise err

            try:
                if head == 0:
                    head_lemma = 'ROOT'
                    head_pos = 'ROOT'
                else:
                    head_lemma = df_copy.iloc[head - 1]['LEMMA']
                    head_pos = df_copy.iloc[head - 1]['XPOS']
            except Exception as err:
                print("==== HEAD HEAD BLAH BLAH ====", 'head', head, type(head))
                print(err)
                raise err

            features_dict['head_lemma_pos'] = f"{head_lemma}_{head_pos}"


            ### extract TOKEN POSITION TO PREDICATE ###
            if i < (pred_id - 1):
                features_dict['token_position'] = 'before'
            elif i > (pred_id-1):
                features_dict['token_position'] = 'after'
            else:
                features_dict['token_position'] = 'same'

        
            ### extract LOCATION OF GOVERNING CATEGORY 'nsubj, dobj', and its position to predicate ###
            if df_row['DEPREL'] == 'nsubj':
                
                if i < (pred_id - 1):
                    token_position = 'nsubj_before'
                elif i > (pred_id-1):
                    token_position = 'nsubj_after'
                else:
                    continue
            
            elif df_row['DEPREL'] == 'dobj':
                if i < (pred_id - 1):
                    token_position = 'dobj_before'
                elif i > (pred_id-1):
                    token_position = 'dobj_after'
                else:
                    continue
            else:
                token_position = '0'

            features_dict['gov_cat_position'] = token_position


            ### extract PREDICATE & HEAD OF TOKEN ###
            
            row_head = df_row['HEAD']
            if row_head == 0:
                head_form = 0
            else:
                head_rows = df_copy.loc[df_copy['ID'] == row_head,'FORM']
                if not head_rows.empty:
                    head_form = head_rows.values[0] 
                else:
                    head_form = 0   
            features_dict['pred_head']= f"{pred_form}_{head_form}"



            # -------------------------------------------------------------
            # -------------------------------------------------------------

            # all features_dict are appended to features_dict_list:
            features_dict_list.append(features_dict) 
        
    return features_dict_list

