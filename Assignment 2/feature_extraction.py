import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def recursive_find_path(df_copy, i):
    '''
    This function is used to find a full path from the current token to the root of the sentence.

    :param DataFrame df_copy: subset of the current sentence in the data
    :param int i: token's position within the sentence

    :return list: a list containing all the dependency tags of tokens in the path from the current token to the root
    '''
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



def get_features(file_path, dataset_type):
    '''
    This function creates a new file with the columns for following features:
    - each token's voice (based on dependency relation tags) + position to predicate,
    - predicate's lemma + its pos tag (same for all tokens within a sentence)
    - each token's head word - its lemma + pos tag
    - each token's position to predicate
    - governing category, 'nsubj, dobj' - marking each token with these tags + position to predicate
    - each token's predicate + head of token
    - path from the token to the root, a list of strings (DEP tag)

    :param str file_path: path to the file
    :param str dataset_type: either 'train' or 'test'

    :return: None
    '''
    #read in data to a pandas dataframe
    train_df = pd.read_csv(file_path, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')


    # change values in columns 'Sent_ID, Copy_ID, ID' to integer
    train_df = train_df.astype({'Sent_ID':'int'})
    train_df = train_df.astype({'Copy_ID':'int'})
    train_df = train_df.astype({'ID':'int'})
    
    # make a dataframe to save additional features into
    features_df = train_df.copy()
    # index 
    row_index = 0

    copy_id_list = train_df['Copy_ID'].unique()

    for num in copy_id_list:
        df_copy = train_df.loc[train_df['Copy_ID'] == num] # subset df for each sentence
        
        # get the length of sentence
        max_wds_count = df_copy['ID'].max()


        # find ID value of the predicate
        pred_item = df_copy['ID'][df_copy['UP:PRED'].str.len() > 2]
        pred_id = 0
        
        # predicate for feature "PREDICATE & HEAD OF TOKEN"
        if not pred_item.empty:
            pred_id = pred_item.values[0]
            pred_form = df_copy.loc[df_copy['ID'] == pred_id,'FORM'].values[0]
        else:
            pred_form = 0

        # recursively find path
        # extract features within each sentence boundary
        for i in range(max_wds_count):            
            # row 'i'
            df_row = df_copy.iloc[i]


            # COMMENT OUT ANY FEATURES THAT ARE NOT NEEDED BELOW
            # -------------------------------------------------------------
            # -------------------------------------------------------------


            ### extract VOICE + POSITION TO PREDICATE ###
            if df_row['DEPREL'] == 'nsubj:pass':
                if pred_id == 0:
                    features_df.loc[row_index,['voice_position-to-pred']] = 0
                elif i < (pred_id - 1):
                    features_df.loc[row_index,['voice_position-to-pred']] = '1_before'
                elif i > (pred_id-1):
                    features_df.loc[row_index,['voice_position-to-pred']] = '1_after'
                else:
                    features_df.loc[row_index,['voice_position-to-pred']] = '1_same'
            else:
                if pred_id == 0:
                    features_df.loc[row_index,['voice_position-to-pred']] = 0
                elif i < (pred_id-1):
                    features_df.loc[row_index,['voice_position-to-pred']] = '0_before'
                elif i > (pred_id-1):
                    features_df.loc[row_index,['voice_position-to-pred']] = '0_after'
                else:
                    features_df.loc[row_index,['voice_position-to-pred']] = '0_same'


            ### extract PREDICATE LEMMA + POS TAG ###
            if i == (pred_id - 1):
                features_df.loc[row_index,['pred-lemma_pos']] = f"{df_copy.iloc[pred_id-1]['LEMMA']}_{df_copy.iloc[pred_id-1]['XPOS']}"
            else:
                features_df.loc[row_index,['pred-lemma_pos']] = '0'


            ### #extract HEAD WD OF TOKEN + POS TAG ###
            head = int(df_row['HEAD'])
        
            if head == 0:
                head_lemma = 'ROOT'
                head_pos = 'ROOT'
            else:
                head_lemma = df_copy.iloc[head - 1]['LEMMA']
                head_pos = df_copy.iloc[head - 1]['XPOS']

            features_df.loc[row_index,['head_lemma_pos']] = f"{head_lemma}_{head_pos}"

            
            ### extract TOKEN POSITION TO PREDICATE ###
            if pred_id == 0:
                features_df.loc[row_index,['token_position']] = 0
            elif i < (pred_id - 1):
                features_df.loc[row_index,['token_position']] = 'before'
            elif i > (pred_id-1):
                features_df.loc[row_index,['token_position']] = 'after'
            else:
                features_df.loc[row_index,['token_position']] = 'same'

            
            ### extract LOCATION OF GOVERNING CATEGORY 'nsubj, dobj', and its position to predicate ###
            if df_row['DEPREL'] == 'nsubj':
                if pred_id == 0:
                    token_position = 0
                elif i < (pred_id - 1):
                    token_position = 'nsubj_before'
                elif i > (pred_id-1):
                    token_position = 'nsubj_after'
                else:
                    token_position = 'nsubj_same'
            
            elif df_row['DEPREL'] == 'dobj':
                if pred_id == 0:
                    token_position = 0
                elif i < (pred_id - 1):
                    token_position = 'dobj_before'
                elif i > (pred_id-1):
                    token_position = 'dobj_after'
                else:
                    token_position = 'dobj_same'
            else:
                token_position = '0'
            
            features_df.loc[row_index,['gov_cat_position']] = token_position

            
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
            features_df.loc[row_index,['pred_head']] = f"{pred_form}_{head_form}"

            
            ### extract PATH OF TOKEN TO ROOT, IN DEP TAG ###
            
            sent_path_list = recursive_find_path(df_copy, i)
            sent_path = ':'.join(sent_path_list)
            features_df.loc[row_index,['path']] = sent_path


            row_index +=1
            ###########################################

            # -------------------------------------------------------------
            # -------------------------------------------------------------

    # save the dataframe with additional features into a new file
    features_df.to_csv(f'../data/{dataset_type}_with_features.tsv',sep='\t',index=False,encoding='utf-8',quotechar='№')

def extract_features(filepath, task_name, data_type):
    '''
    Function for extracting features:
    - each token, 
    - each token's pos tag
    - each token's voice (based on dependency relation tags) + position to predicate,
    - predicate's lemma + its pos tag (same for all tokens within a sentence)
    - each token's head word - its lemma + pos tag
    - each token's position to predicate
    - governing category, 'nsubj, dobj' - marking each token with these tags + position to predicate
    - each token's predicate + head of token
    - path from the token to the root, a list of strings (DEP tag)

    :param str filepath: path to the file with all the features
    :param str task_name: 'AC' or 'AI' 
    :param str data_type: 'train' or 'test'

    :return list features_list: list of dictionaries of features
    '''
    features_df = pd.read_csv(filepath, sep='\t', header=0, encoding='utf-8', quotechar='№', engine='python')
    features_list = []
    if (task_name == 'AC') and (data_type == 'train'):
        features_df = features_df.loc[features_df['UP:ARGHEADS'] != 'O']
    elif (task_name == 'AC') and (data_type == 'test'):
        features_df = features_df.loc[features_df['Predictions_AI'] != 'O']
    for i, row in features_df.iterrows():
        feature_dict = {}
        # extract token
        feature_dict['token'] = row['FORM']
        # extract pos
        feature_dict['pos'] = row['XPOS']
        # extract voice + position to predicate
        feature_dict['voice_position-to-pred'] = row['voice_position-to-pred']
        # extract predicate lemma + predicate pos
        feature_dict['pred_lemma_pos'] = row['pred-lemma_pos']
        # extract head lemma + head pos
        feature_dict['head_lemma_pos'] = row['head_lemma_pos']
        # extract position to predicate
        feature_dict['token_position'] = row['token_position']
        # extract governing category + position to predicate
        feature_dict['gov_cat_position'] = row['gov_cat_position']
        # extract predicate + head token
        feature_dict['pred_head'] = row['pred_head']
        # extract path from token to root 
        feature_dict['path'] = row['path']

        features_list.append(feature_dict)

    return features_list
