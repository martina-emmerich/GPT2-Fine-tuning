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


from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, set_seed,  BitsAndBytesConfig

from peft import LoraConfig

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

    #Model generation arguments for generation during evaluation
    parser.add_argument('--max-output-length', default=200, help='maximum length of output sequence during evaluation' )
    parser.add_argument('--top-k', default=50, type=int, help='number of highest probability vocabulary tokens to keep for top-k-filtering')
    #only the set of tokens with proability that add up to top_p or higher are used for generation
    parser.add_argument('--top-p', default=0.95, type=float, help='probability threshold to select the set of tokens used for generation')

    #Model arguments 
    parser.add_argument('--model', default='gpt2', help='model name from HuggingFace')
    parser.add_argument('--qlora', default=False, type=bool, help='training using QLORA peft method instead of full fine-tuning')
    
    #QLORA config arguments
    parser.add_argument('--rank_lora', default=64, type=int, help='rank of lo rank matrices in LORA')
    parser.add_argument('--alpha_lora', default=16, type=int, help='alpha scaling parameter in LORA')
    parser.add_argument('--targets_lora', default ='c_attn', type=str, nargs='+', help='list of modules to apply adapters to for LORA')

    #Optimization arguments
    parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
    parser.add_argument('--sample-every', default=100, type=int, help='every number of steps after which a random sample is outputted')
    parser.add_argument('--epochs', default=4, type=int, help='train until specified epoch')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='Adam’s epsilon for numerical stability')

    #Saving and loading checkpoint arguments
    parser.add_argument('--check-dir', default='Checkpoints', help='path to directory to save checkpoints')
    parser.add_argument('--restore-file', type=str, help='filename/directory name to load checkpoint') #change name this is not the right one
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--load-checkpoint', default=False, type=bool, help='whether to load the model from checkpoint')


    #Save model arguments
    parser.add_argument('--output-dir', default='GPT2_fine_tuned_Austen', help='path to save logs')

    args = parser.parse_args()
    return args


def validate(val_dataset, model, args, epoch):
    """
    Loads validation data and runs validation loop.
    Returns the average validation loss and the amount of time it took to evaluate the model on the validation set. 
    """
    #Set all seeds to allow for reproducible runs
    torch.manual_seed(64) 
    np.random.seed(64) 
    torch.manual_seed(64)

    val_dataloader = DataLoader(val_dataset,
                            sampler=SequentialSampler(val_dataset),
                           batch_size=args.batch_size) 

    t0 = time.time()
    model.eval()

    total_eval_loss = 0

    # display progress
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

    print('Commencing training!')
    #set all seeds
    random.seed(64)
    torch.manual_seed(64) #42
    np.random.seed(64) #42
    torch.cuda.manual_seed(64)
    # set seed before initializing model.
    set_seed(64)

   
    #instantiate tensorboard logger
    writer = SummaryWriter()
   
    print('Arguments: {}'.format(vars(args)))

    #load model tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model, 
                                          bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>')

    #load dataset
    with open(args.data, 'rb') as f:
        data=pickle.load(f)
        print('Total length of data loaded', len(data))
    dataset =  AustenDataset(data, tokenizer, max_length=args.max_input_length)
    print('Number of input sequences after tokenization', len(dataset))

    # split data into train and validation sets
    train_size = int(args.train_size*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Number of samples for training =", train_size)
    print("Number of samples for validation =", val_size)
    

    train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=args.batch_size) 


    #load base model or base model with qlora adapters
    if args.qlora:
        print(f'Loading {args.model} quantized and with lora adapaters.')
        model = utils.load_qloramodel(args.model, tokenizer, args)
    else:
        print(f'Training all model parameters for model {args.model}')
        model = utils.load_basemodel(args.model, DEVICE, tokenizer)

    
    # using AdamW optimizer with default parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # total training steps is the number of data points times the number of epochs
    total_training_steps = len(train_dataloader)*(args.epochs) 

    #print(f'len of training data loader {len(train_dataloader)}')
    #print(next(iter(train_dataloader)))
    print(f'Total training steps {total_training_steps}.')

    # setting a variable learning rate using scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=total_training_steps)
    
    #load checkpoint if needed
    if args.load_checkpoint:
        print(f'Loading model, optimizer and scheduler from checkpoint, from following dir: {args.restore_file}')
        path_to_checkpoint = os.path.join('.', args.check_dir, args.restore_file, 'state_dict')

        # load last checkpoint state dict  to get optimizer and epochs
        state_dict = utils.load_checkpoint(path_to_checkpoint, model, optimizer, scheduler)  
        last_epoch = state_dict['epoch'] 

        print(f'Restarting training from {last_epoch+1}, with learning rate {scheduler._last_lr}.')
    else:
        print('Fine-tuning from base model')
        last_epoch = 0
       
    total_t0 = time.time()

    training_stats = []

    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)/1e+9} GB of memory")
    for epoch in range(last_epoch +1, args.epochs+1): 
        print(f'Beginning epoch {epoch} of {args.epochs}')
        
        t0 = time.time()
        total_train_loss = 0
        model.train()
    
        # display progress
        progress_bar = tqdm(train_dataloader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)
        
        torch.manual_seed(64)
        # iterate over the training set
        for step, batch in enumerate(progress_bar):
            b_input_ids = batch[0].to(DEVICE)
            b_labels = batch[0].to(DEVICE)
            b_masks = batch[1].to(DEVICE)

            model.zero_grad()
            
            outputs = model(b_input_ids,
                    labels=b_labels,
                    attention_mask=b_masks)
    
            loss = outputs[0] #outputs.loss 
    
            batch_loss = loss.item()
            total_train_loss += batch_loss

            #print batch step, batch loss on training data, time elapsed and learning rate every 500 steps
            if step != 0 and step % 500 == 0: 
                elapsed = utils.format_time(time.time()-t0)
                print(f' Step {step} of {len(train_dataloader)}. Loss: {batch_loss}. Time: {elapsed}. Learning rate {scheduler._last_lr}')

            # sample example output every x steps
            if step != 0 and step % args.sample_every == 0:

                elapsed = utils.format_time(time.time()-t0)
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
                for _, sample_output in enumerate(sample_outputs):
                    print(f'Example ouput: {tokenizer.decode(sample_output, skip_special_tokens=True)}')
                print()

                model.train()

            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"gpu used {torch.cuda.max_memory_allocated(device=None)/1e+9} GB of memory")
    

        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = utils.format_time(time.time()-t0)
        writer.add_scalar("Loss/avg_train", avg_train_loss, epoch)
      
        print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
        print()

        avg_val_loss, val_time = validate(val_dataset, model, args, epoch)
        
        print(f'Validation loss: {avg_val_loss}. Validation Time: {val_time}')
        print()
        writer.add_scalar("Loss/avg_val", avg_val_loss, epoch)

        #save model after N epochs
        if epoch % args.save_interval == 0:
            print(f'Saving model after {epoch} epochs.')
            utils.save_model(args.check_dir+'\Epoch_'+str(epoch), model, tokenizer, args, optimizer, epoch, avg_val_loss, scheduler, checkpoint=True)
        # record all statistics from this epoch.
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
   
    main(args)

