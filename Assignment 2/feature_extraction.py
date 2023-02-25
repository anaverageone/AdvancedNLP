import pandas as pd
import numpy as np


def recursive_find_path(df_copy, i):
    current_tag_list = []
 
    df_row = df_copy.iloc[i]
 
    deps = df_row['DEPS']
    deps_tag = deps.split(':',1)[-1]
    current_tag_list.append(deps_tag)

    if df_row['HEAD'] == 0:
        return current_tag_list
    
    next_id = df_row['HEAD'] - 1
    recursive_tag_list = recursive_find_path(df_copy, next_id)
    
    return current_tag_list + recursive_tag_list



def extract_features(file_path):
    '''Function for extracting features,
    - each token, 
    - each token's pos tag
    - each token's voice (based on dependency relation tags) + position to predicate,
    - predicate's lemma + its pos tag (same for all tokens within a sentence)
    - each token's head word - its lemma + pos tag
    - each token's position to predicate
    - governing category, 'nsubj, dobj' - marking each token with these tags + position to predicate
    - each token's predicate + head of token
    - path from the token to the root, a list of strings (DEP tag)
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

        # find ID value of the predicate
        # pred_row = df_copy.loc[df_copy['UP:PRED']!= '_']
        # pred_id = pred_row['ID']
        pred_item = df_copy['ID'][df_copy['UP:PRED'].str.len() > 2]
        pred_id = 0
        # try:
        #     pred_id = pred_item.values[0]
        # except:
        #     print(' !!! WARNING !!! There is no predicate in sentence with copy_id:', num)
        #     continue

        # print("pred_id:",pred_id)
        # print("type pred_id:", type(pred_id))

        # create a new column "VOICE" and set all values to "0"
        df_copy['VOICE'] = '0' 

        # predicate for feature "PREDICATE & HEAD OF TOKEN"
        if not pred_item.empty:
            pred_id = pred_item.values[0]
            pred_form = df_copy.loc[df_copy['ID'] == pred_id,'FORM'].values[0]
        else:
            pred_form = 0

        # recursively find path
        # extract features within each sentence boundary
        for i in range(max_wds_count):
            features_dict = {}
            
            # row 'i'
            df_row = df_copy.iloc[i]

            # COMMENT OUT ANY FEATURES THAT ARE NOT NEEDED BELOW
            # -------------------------------------------------------------
            # -------------------------------------------------------------

            # ### extract each token ###
            features_dict['token']=df_row['FORM']
            

            ### extract POS, ALL TOKENS ###
            features_dict['pos'] = df_row['XPOS']

            ### extract VOICE + POSITION TO PREDICATE ###
            if df_row['DEPREL'] == 'nsubj:pass':
                if pred_id == 0:
                    df_row['VOICE'] = 0
                elif i < (pred_id - 1):
                    df_row['VOICE'] = '1_before'
                elif i > (pred_id-1):
                    df_row['VOICE'] = '1_after'
                else:
                    df_row['VOICE'] = '1_same'
            else:
                if pred_id == 0:
                    df_row['VOICE'] = 0
                elif i < (pred_id-1):
                    df_row['VOICE'] = '0_before'
                elif i > (pred_id-1):
                    df_row['VOICE'] = '0_after'
                else:
                    df_row['VOICE'] = '0_same'

            features_dict['voice_position-to-pred'] = df_row['VOICE']

            
            ### extract PREDICATE LEMMA + POS TAG ###
            if pred_id == 0:
                features_dict['pred-lemma_pos'] = 0
            if i == (pred_id - 1):
                features_dict['pred-lemma_pos'] = f"{df_row['LEMMA']}_{df_row['XPOS']}"

            
            ### #extract HEAD WD OF TOKEN + POS TAG ###
            
            head = int(df_row['HEAD'])
            
    
            if head == 0:
                head_lemma = 'ROOT'
                head_pos = 'ROOT'
            else:
                head_lemma = df_copy.iloc[head - 1]['LEMMA']
                head_pos = df_copy.iloc[head - 1]['XPOS']

            features_dict['head_lemma_pos'] = f"{head_lemma}_{head_pos}"

            
            ### extract TOKEN POSITION TO PREDICATE ###
            if pred_id == 0:
                features_dict['token_position'] = 0
            elif i < (pred_id - 1):
                features_dict['token_position'] = 'before'
            elif i > (pred_id-1):
                features_dict['token_position'] = 'after'
            else:
                features_dict['token_position'] = 'same'

            
            ### extract LOCATION OF GOVERNING CATEGORY 'nsubj, dobj', and its position to predicate ###
            if df_row['DEPREL'] == 'nsubj':
                if pred_id == 0:
                    token_position = 0
                elif i < (pred_id - 1):
                    token_position = 'nsubj_before'
                elif i > (pred_id-1):
                    token_position = 'nsubj_after'
                else:
                    continue
            
            elif df_row['DEPREL'] == 'dobj':
                if pred_id == 0:
                    token_position = 0
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

            
            ### extract PATH OF TOKEN TO ROOT, IN DEP TAG ###
            
            sent_path = recursive_find_path(df_copy, i)
            features_dict['path'] = sent_path
            
            ###########################################

            # -------------------------------------------------------------
            # -------------------------------------------------------------

            # all features_dict are appended to features_dict_list:
            features_dict_list.append(features_dict)

        
    return features_dict_list

