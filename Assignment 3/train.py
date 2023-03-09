import random, time, os
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
from transformers import BertTokenizer
import pandas as pd

# Our code behind the scenes!
import bert_utils_srl as utils
import make_jsonl as jsonl

# ------------------------------------------------------------------- #
# Initialize Hyperparameters
# ------------------------------------------------------------------- #


EPOCHS = 2

BERT_MODEL_NAME = "bert-base-multilingual-cased"
GPU_RUN_IX=0

SEED_VAL = 1234500
SEQ_MAX_LEN = 512 #256
PRINT_INFO_EVERY = 10 # Print status only every X batches
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4

SAVE_MODEL_DIR = "saved_models/MY_BERT_SRL/"

LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
PRED_FILENAME = f"{SAVE_MODEL_DIR}/pred2index.json"

LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

if not os.path.exists(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)


# Initialize Random seeds and validate if there's a GPU available...
device, USE_CUDA = utils.get_torch_device(GPU_RUN_IX)
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)



# ------------------------------------------------------------------- #
# Set up log file location for recording progress
# ------------------------------------------------------------------- #

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=f"{SAVE_MODEL_DIR}/BERT_TokenClassifier_train_{EPOCHS}.log")
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
logging.info("Start Logging")






def main(argv=None):
    '''
    This function provides the following:
    - load the training set & dev set
    - pre-process original datasets into jsonl format
    - pre-process jsonl format datasets with BertTokenizer
    - datasets formatted to DataLoader matrices
    - fine-tune the model with training set, using model "bert-base-multilingual-cased"
    - evaluate trained model with dev set
    - trained model file is saved for future use
    - all progress are recorded in log files
    '''
    
    if argv == None:
        argv = sys.argv
    
    split_sentences = argv[1]
    create_jsonl = argv[2]

    # ------------------------------------------------------------------- #
    print("------ Read dataset and convert to jsonl format ------")
    # ------------------------------------------------------------------- #

    train_jsonl_file, test_jsonl_file = jsonl.main(['make_jsonl.py', False, True])
    ### ------ update training data jsonl file ------ ###
    TRAIN_DATA_PATH = train_jsonl_file 
    ### ------ update testing data jsonl file ------ ###
    DEV_DATA_PATH =  test_jsonl_file 

        
    # ------------------------------------------------------------------- #
    # Initialize Tokenizer
    # ------------------------------------------------------------------- #

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

    # ------------------------------------------------------------------- #
    print("------ Train Dataset - load jsonl training set, pre-process with tokenizer for creating DataLoader ------")
    # ------------------------------------------------------------------- #
    train_data, train_labels, train_label2index, train_pred_sense, train_pred2index = utils.read_json_srl(TRAIN_DATA_PATH) #, has_labels=True)
    train_inputs, train_masks, train_labels, seq_lengths, train_pred = utils.data_to_tensors(train_data, 
                                                                                tokenizer, 
                                                                                max_len=SEQ_MAX_LEN, 
                                                                                labels=train_labels, 
                                                                                label2index=train_label2index,
                                                                                pred_sense=train_pred_sense,
                                                                                pred2index=train_pred2index,
                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID)


    utils.save_label_dict(train_label2index, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in train_label2index.items()} 

 
    # ------------------------------------------------------------------- #
    print("------ Train Dataset - create DataLoader ------")
    # ------------------------------------------------------------------- #
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_pred)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)


    # ------------------------------------------------------------------- #
    print("------ Dev Dataset - load jsonl dev set, pre-process with tokenizer, create DataLoader ------")
    # ------------------------------------------------------------------- #
    dev_data, dev_labels, _ , dev_pred_sense, _ = utils.read_json_srl(DEV_DATA_PATH) #, has_labels=True)
    dev_inputs, dev_masks, dev_labels, dev_lens, dev_pred = utils.data_to_tensors(dev_data, 
                                                                        tokenizer, 
                                                                        max_len=SEQ_MAX_LEN, 
                                                                        labels=dev_labels, 
                                                                        label2index=train_label2index,
                                                                        pred_sense=dev_pred_sense,
                                                                        pred2index=train_pred2index,
                                                                        pad_token_label_id=PAD_TOKEN_LABEL_ID)  

    # Create the DataLoader for our Development set.
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_pred) 
    dev_sampler = RandomSampler(dev_data)     
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)


    # ------------------------------------------------------------------- #
    print("------ Initialize model components ------")
    # ------------------------------------------------------------------- #
    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(train_label2index))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = index2label  
    model.config.label2id = train_label2index  
    if USE_CUDA: model.cuda()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


    # ------------------------------------------------------------------- #
    print("------ Start fine-tuning cycles ------")
    # ------------------------------------------------------------------- #

    loss_trn_values, loss_dev_values = [], []

    for epoch_i in range(1, EPOCHS+1):
        # Perform one full pass over the training set.
        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        logging.info('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_preds = batch[3].to(device) ###### added predicate ######

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, token_type_ids=b_preds) #input_ids = 
            ### added token_tupe_ids = b_preds, also 'input_ids')
            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(utils.format_time(time.time() - t0)))

        # ------------------------------------------------------------------- #
        print("------ Evaluate trained model with dev set ------")
        # ------------------------------------------------------------------- #
        # After the completion of each training epoch, measure our performance on our validation set.
        t0 = time.time()
        results, preds_list = utils.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label, PAD_TOKEN_LABEL_ID, prefix="Validation Set")
        
        loss_dev_values.append(results['loss'])
        logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
        logging.info("  Validation took: {:}".format(utils.format_time(time.time() - t0)))


        # Save Checkpoint for this Epoch
        utils.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)
        print("------ Model file saved ------")


    utils.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    utils.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    logging.info("")
    logging.info("------ Training complete! ------")

if __name__ == '__main__':
    my_args = ['make_jsonl.py', False, True]
    main(my_args)