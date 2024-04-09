from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config 

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE:', DEVICE)
model = 'gpt2-medium'

tokenizer = GPT2Tokenizer.from_pretrained(model) 
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained(model)
model = GPT2LMHeadModel.from_pretrained(model, config=config)
model = model.to(DEVICE)

tot_params = sum(p.numel() for p in model.parameters())
print(f'Model with {tot_params:,} number of parameters')

prompt = "in the garden" # dancing on the balcony

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(DEVICE)
print(generated)

sample_outputs = model.generate(
                                generated, 
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300, 
                                top_p=0.95, 
                                num_return_sequences=3 
                                )

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))