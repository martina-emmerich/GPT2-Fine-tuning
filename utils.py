import datetime
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round(elapsed))))

def save_model(output_dir, model, tokenizer, args =None, optimizer = None, epoch=None, val_loss=None, lr_scheduler=None,checkpoint=False):
    os.makedirs(output_dir, exist_ok=True)
    #save model state dictionary with all infomration needed to continue training from the checkpoint
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
        #save last model and tokenizer 
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def load_checkpoint(path, model, optimizer, scheduler): 
    #load checkpoint information and model to continue training 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    return checkpoint

def load_basemodel(model_name, device, tokenizer):
    #load base model from HuggingFace
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    #set model embedding length
    model.resize_token_embeddings(len(tokenizer))

    #running the model on GPU
    model = model.to(device)

    return model

def print_trainable_parameters(model):
    """
    Prints the names of the trainable parameters in the model.
    """
   # params = torch.nn.ParameterList()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def load_qloramodel(model_name, device, tokenizer, args):
    #load 4-bit quantized model
    model = GPT2LMHeadModel.from_pretrained(
        model_name,    
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16)
     # set model embedding length
    model.resize_token_embeddings(len(tokenizer))

    # add LoRA adapters to model
    model = prepare_model_for_kbit_training(model)
    #for name, param in model.named_parameters():
     #   print(name)
    
    lora_config = LoraConfig(
                r=64, 
                lora_alpha=16, 
               # target_modules = ['c_proj'],
                lora_dropout=0.1, 
                bias="none", 
                modules_to_save = ["lm_head", "embed_tokens"],        # needed because we added new tokens to tokenizer/model
                task_type="CAUSAL_LM")
    
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    print_trainable_parameters(model)

    return model
