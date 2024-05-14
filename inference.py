from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import torch
import os

def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('GPT2 Model fine-tuned on Austen dataset')

    #Add arguments to load model for inference
    parser.add_argument('--base_model', default='gpt2-medium', type=str, help='name of the initial model before qlora fine-tuning, needed only if loading a qlora model')
    parser.add_argument('--model_path', default=None, type=str, help='path to fine-tuned local model directory')
    parser.add_argument('--prompt', default=None, type=str, help='prompt for the model to complete')
    parser.add_argument('--qlora', default=False, type=bool, help='whether the model we are loading is saved as qlora adapters')
    
    #Add arguments for generate 
    parser.add_argument('--max_length', default=300, type=int, help='maximum length of generated sequence')
    parser.add_argument('--top_p', default=0.95, type=float, help='probability threshold to select the set of tokens used for generation')
    parser.add_argument('--n_return_seqs', default=3, type=int, help='number of sequences the model should generate for each prompt')
    parser.add_argument('--top-k', default=50, type=int, help='number of highest probability vocabulary tokens to keep for top-k-filtering')

    args = parser.parse_args()
    return args

def load_qlora_model(adapter_path, tokenizer, base_model):

    model = AutoModelForCausalLM.from_pretrained(base_model,  quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"), torch_dtype=torch.bfloat16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)

    return model

def load_model(model, tokenizer):

    config = GPT2Config.from_pretrained(model)
    model = GPT2LMHeadModel.from_pretrained(model, config=config)
    model.resize_token_embeddings(len(tokenizer))

    return model  

def main(args):
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    print(len(tokenizer))
    if args.qlora:
        print('Loading qlora fine-tuned model.')
        model = load_qlora_model(args.model_path, tokenizer, args.base_model)
        print(f"Loaded the QLoRA fine-tuned version of {model.config._name_or_path}.")
    else:
        print('Loading fine-tuned model.')
        model = load_model(args.model_path, tokenizer) 
        print(f"Loaded the fine-tuned version of {model.config._name_or_path}.")
    model.to(DEVICE)

    prompt = "<|startoftext|>" + args.prompt 

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(DEVICE)
    print(generated)

    sample_outputs = model.generate(
                                generated, 
                                do_sample=True,   
                                top_k=args.top_k, 
                                max_length = args.max_length, 
                                top_p=args.top_p,  
                                num_return_sequences=args.n_return_seqs 
                                )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if __name__=='__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda:0" 
    print('DEVICE:', DEVICE)
    args=get_args()
   
    main(args)