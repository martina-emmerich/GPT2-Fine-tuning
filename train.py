import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import random
import time
import datetime
import os
import logging
import argparse
from tqdm import tqdm


from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, set_seed

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler

from dataset import AustenDataset
import utils


def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('GPT2 Model fine-tuned on Austen dataset')

    #Add arguments for data options
    parser.add_argument('--data', default='austen_dataset.pkl', help='path to data directory')
    parser.add_argument('--batch-size', default=8, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-input-length', default=100, type=int, help='maximum length of data input sequence' )
    parser.add_argument('--train-size', default=0.9, type=float, help='percentage of data for training split')

    #Add model generation arguments for generation during evaluation
    parser.add_argument('--max-output-length', default=200, help='maximum length of output sequence during evaluation' )
    parser.add_argument('--top-k', default=50, type=int, help='number of highest probability vocabulary tokens to keep for top-k-filtering')
    #only the set of tokens with proability that add up to top_p or higher are used for generation
    parser.add_argument('--top-p', default=0.95, type=float, help='probability threshold to select the set of tokens used for generation')
    #parser.add_argument('--sampling', default=True, type=bool, help='whether or not to use sampling, if set to false uses greedy decoding') #we always wantit true otherwise passing topk and topp will generate an error
    # do sample? whether or not to do sampling, in huggingface generate i defaults to false but we want it to be true 
    #bos token id (why is it random.randint(1, 30 000)? becaue we are passing the model a random prompt to generate out random smaple
    #num_return_sequences should be always set to one in generate I suppose 

    #Add model arguments 
    parser.add_argument('--model', default='gpt2', help='model name from HuggingFace')
    
    #Add optimization arguments
    parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
    parser.add_argument('--sample-every', default=100, type=int, help='every number of steps after which a random sample is outputted')
    parser.add_argument('--epochs', default=4, type=int, help='train until specified epoch')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='Adam’s epsilon for numerical stability')

    #Add checkpoint arguments (to do)
   # parser.add_argument('--log-file', default=None, help='path to save logs') #implement logging?
    parser.add_argument('--check-dir', default='Checkpoints', help='path to directory to save checkpoints')
    parser.add_argument('--restore-file', type=str, help='filename/directory name to load checkpoint') #change name this is not the right one
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
   # parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')
    parser.add_argument('--load-checkpoint', default=False, type=bool, help='whether to load the model from checkpoint')
    #parse arguments

    #Save model arguments
    parser.add_argument('--output-dir', default='GPT2_fine_tuned_Austen', help='path to save logs')

    args = parser.parse_args()
    return args


def validate(val_dataset, model, args, epoch):
    torch.manual_seed(64) #42
    np.random.seed(64) #42
    torch.manual_seed(64)
    val_dataloader = DataLoader(val_dataset,
                            sampler=SequentialSampler(val_dataset),
                           batch_size=args.batch_size) 

    t0 = time.time()
    model.eval()

    total_eval_loss = 0
    #nb_eval_steps = 0

    # Display progress
    progress_bar_eval = tqdm(val_dataloader, desc='| Epoch validation {:03d}'.format(epoch), leave=False, disable=False)
    torch.manual_seed(64)
    for batch in progress_bar_eval:
        b_input_ids = batch[0].to(DEVICE)
        b_labels = batch[0].to(DEVICE)
        b_masks = batch[1].to(DEVICE)
        
        with torch.no_grad():        
            
            outputs  = model(b_input_ids,  
                         attention_mask = b_masks,
                         labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss   

    avg_val_loss = total_eval_loss / len(val_dataloader)  
    val_time = utils.format_time(time.time() - t0)    

    return avg_val_loss, val_time



def main(args):
    """
    Main training function. Trains the selected model (GPT2) over several epochs, batching the training data and using AdamW optimizer
    for the learning rate schedule.
    """
   # logging.info('Commencing training!')
    print('Commencing training!')
    torch.manual_seed(64) #42
    np.random.seed(64) #42
    torch.cuda.manual_seed(64)
    # Set seed before initializing model.
    set_seed(64)#training_args.seed)

    #if use file to log too add some checks to make sure the files passed to arguments exist
    #instantiate tensorboard logger
    writer = SummaryWriter()
   # logging.info('Arguments: {}'.format(vars(args)))
    print('Arguments: {}'.format(vars(args)))

    #Load model tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model, 
                                          bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>')

    #Load datasets
   # def load_data():
    with open(args.data, 'rb') as f:
        data=pickle.load(f)
        print('Total length of data loaded', len(data))
    dataset =  AustenDataset(data, tokenizer, max_length=args.max_input_length)
    print('Number of input sequences after tokenization', len(dataset.input_ids), len(dataset))
    # Split data into train and validation sets
    train_size = int(args.train_size*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Number of samples for training =", train_size)
    print("Number of samples for validation =", val_size)
    

    train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=args.batch_size) 

    # Load model configuration, instance on GPU either from checkpoint or HUggingFace
    #Also set up optimizer and scheduler

    #Load base model
    model = utils.load_basemodel(args.model, DEVICE, tokenizer)

    
    # Using AdamW optimizer with default parameters
    # Using AdamW optimizer with default parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)#AdamW(model.parameters(), lr=args.lr, eps=args.eps)
   # print(optimizer)

    # Total training steps is the number of data points times the number of epochs
    total_training_steps = len(train_dataloader)*(args.epochs) # - last_epoch) #subtract the numbe rof epochs already run when training from a checkpoint, otherwise last_epoch is 0
    #print(f'len of training data loader {len(train_dataloader)}')
    #print(next(iter(train_dataloader)))
    print(f'Total training steps {total_training_steps}.')

    # Setting a variable learning rate using scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=total_training_steps)
    
    #Load checkpoint if needed
    if args.load_checkpoint:
        print(f'Loading model, optimizer and scheduler from checkpoint, from following dir: {args.restore_file}')
        path_to_checkpoint = os.path.join('.\\', args.check_dir, args.restore_file, 'state_dict')
       # print(path_to_checkpoint)
      #  model = utils.load_basemodel(args.model, DEVICE, tokenizer)
        #Scheduling Optimizer
        # Using AdamW optimizer with default parameters
      #  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)#AdamW(model.parameters(), lr=args.lr, eps=args.eps)
       # print(optimizer)
        # Load last checkpoint state dict if one exists to get optimizer and epochs
        state_dict = utils.load_checkpoint(path_to_checkpoint, model, optimizer, scheduler)  # lr_scheduler
        last_epoch = state_dict['epoch'] #if state_dict is not None else -1
        #print(optimizer)
       # scheduler = state_dict['lr_scheduler']
        print(f'Restarting training from {last_epoch+1}, with learning rate {scheduler}.')
    else:
        print('Loaded a base model from HuggingFace.')
        #model = utils.load_basemodel(args.model, DEVICE, tokenizer)
        #Scheduling Optimizer
        # Using AdamW optimizer with default parameters
       # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)#AdamW(model.parameters(), lr=args.lr, eps=args.eps)
        last_epoch = 0
       
    total_t0 = time.time()

    training_stats = []
    
    for epoch in range(last_epoch +1, args.epochs+1): #(1, args.epochs):
        print(f'Beginning epoch {epoch} of {args.epochs}')
        #implement a stats ordereddictionary to keep track of learning rate, and losses and batch size?
        t0 = time.time()
        total_train_loss = 0
        model.train()
    
        # Display progress
        progress_bar = tqdm(train_dataloader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)
        
        torch.manual_seed(64)
        # Iterate over the training set
        for step, batch in enumerate(progress_bar):
            b_input_ids = batch[0].to(DEVICE)
            b_labels = batch[0].to(DEVICE)
            b_masks = batch[1].to(DEVICE)

            model.zero_grad()
           #I thinkit should model.train here not model.zerograd model.train()
            #from huggingface documentation: The outputs object is a SequenceClassifierOutput, as we can see in the documentation of that class below, it 
            #means it has an optional loss, a logits, an optional hidden_states and an optional attentions attribute. Here we have the loss since we
            #passed along labels, but we don’t have hidden_states and attentions because we didn’t pass output_hidden_states=True or output_attentions=True.
            outputs = model(b_input_ids,
                    labels=b_labels,
                    attention_mask=b_masks)
    
            loss = outputs[0] #outputs.loss 
    
            batch_loss = loss.item()
            total_train_loss += batch_loss
            #printing batch step, batch loss on training data and time elapsed every 500 steps
            if step != 0 and step % 500 == 0:
                elapsed = utils.format_time(time.time()-t0)
                print(f' Step {step} of {len(train_dataloader)}. Loss: {batch_loss}. Time: {elapsed}')

            # Sampling example output every x steps
            if step != 0 and step % args.sample_every == 0:

                elapsed = utils.format_time(time.time()-t0)
               # print(f'Batch {step} of {len(train_dataloader)}. Loss: {batch_loss}. Time: {elapsed}')
                print(f'Generating example output... at step {step} of epoch {epoch}.')

                model.eval()

                sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=args.top_k, 
                                    max_length = args.max_output_length,
                                    top_p=args.top_p, 
                                    num_return_sequences=1
                                )
                for i, sample_output in enumerate(sample_outputs):
                    print(f'Example ouput: {tokenizer.decode(sample_output, skip_special_tokens=True)}')
                print()

                model.train()

            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = utils.format_time(time.time()-t0)
        writer.add_scalar("Loss/avg_train", avg_train_loss, epoch)
       # logging.info(
       # 'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        #' | valid_perplexity {:.3g}'.format(perplexity)) #implement logging for validation and trianing loss
        print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
        print()

        avg_val_loss, val_time = validate(val_dataset, model, args, epoch)
        
        print(f'Validation loss: {avg_val_loss}. Validation Time: {val_time}')
        print()
        writer.add_scalar("Loss/avg_val", avg_val_loss, epoch)

         #Saving model after N epochs
        if epoch % args.save_interval == 0:
            print(f'Saving model after {epoch} epochs.')
            utils.save_model(args.check_dir+'\Epoch_'+str(epoch), model, tokenizer, args, optimizer, epoch, avg_val_loss, scheduler, checkpoint=True)
        # Record all statistics from this epoch.
        training_stats.append(
        {
          'epoch': epoch,
          'Training Loss': avg_train_loss,
          'Valid. Loss': avg_val_loss,
          'Training Time': training_time,
          'Validation Time': val_time,
          'lr_scheduler': scheduler
        })
        print("------------------------------")
    writer.flush()
    print(f'Total training took {utils.format_time(time.time()-total_t0)}')
    print(training_stats)
    utils.save_model(args.output_dir, model, tokenizer)
    print('FINISHED TRAINING SUCCESSFULLY!')
    
if __name__=='__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda:0" 
    print('DEVICE:', DEVICE)
    args=get_args()
    ##args.device_id= 0

    #set up logging to file?
    

    main(args)

