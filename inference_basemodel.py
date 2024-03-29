from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config 

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE:', DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #, bos_token='<|startoftext|>', 
                                        #  eos_token='<|endoftext|>', 
                                         # pad_token='<|pad|>')
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
model = model.to(DEVICE)

#model.eval()

prompt = "people are evil" #'what is ai?'

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(DEVICE)
print(generated)

sample_outputs = model.generate(
                                generated, 
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300, #300
                                top_p=0.95, #0.95 
                                num_return_sequences=3 #3
                                )

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))