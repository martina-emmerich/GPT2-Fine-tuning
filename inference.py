from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config 

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE:', DEVICE)

#model = '.\\bestmodel_run_long_input\Checkpoints_Epoch_2'
model = '.\\bestmodel_run_fasttokenizer_gpt2\Checkpoints_Epoch_3'


tokenizer = GPT2Tokenizer.from_pretrained(model)

config = GPT2Config.from_pretrained(model)
model = GPT2LMHeadModel.from_pretrained(model, config=config)
model = model.to(DEVICE)

prompt = "<|startoftext|> in the garden" #  dancing on the balcony

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