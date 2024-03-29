import datetime
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round(elapsed))))

def save_model(output_dir, model, tokenizer, args =None, optimizer = None, epoch=None, val_loss=None, lr_scheduler=None,checkpoint=False):
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint: 
        state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': args,
        'lr_scheduler': lr_scheduler.state_dict()
        }
        torch.save(state_dict, os.path.join(output_dir, 'state_dict'))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:    
        #model_to_save = model.module if hasattr(model, 'module') else model
       # model_to_save.save_model(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def load_checkpoint(path, model, optimizer, scheduler): #check this works and change if I need it to work differentely
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    return checkpoint

def load_basemodel(model_name, device, tokenizer):
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    # Set model embedding length
    model.resize_token_embeddings(len(tokenizer))

    # Running the model on GPU
    model = model.to(device)

    return model