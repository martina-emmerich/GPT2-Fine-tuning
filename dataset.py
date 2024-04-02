import torch

from torch.utils.data import Dataset



class AustenDataset(Dataset):
  def __init__(self, sentences, tokenizer, max_length=768, gpt2_type="gpt2"):
    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for sentence in sentences:

      #encodings_dict = tokenizer("<|startoftext|>"+sentence["sentence"]+"<|endoftext|>",
                                # truncation=True,
                                # max_length=max_length,
                                # padding="max_length")
      
      encodings_dict = tokenizer("<|startoftext|>"+sentence["sentence"]+"<|endoftext|>",
                                 truncation=True,
                                 max_length=max_length,
                                 return_overflowing_tokens=True,
                                 return_length= True,
                                 padding="max_length")
      #for length, input_ids, mask in zip(encodings_dict["length"], encodings_dict["input_ids"], encodings_dict['attention_mask']):
       # if length == max_length_length:
        #    input_batch.append(input_ids)
      print(encodings_dict)
      print(f"Input IDs length: {len(encodings_dict['input_ids'])}")
      print(f"Input chunk lengths: {(encodings_dict['length'])}")
      print(f"Chunk mapping: {encodings_dict['overflowing_tokens']}") 
      print('num truncated tokens', {encodings_dict['num_truncated_tokens']}) 

      self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
      self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))
      if encodings_dict['num_truncated_tokens'] <= max_length and encodings_dict['num_truncated_tokens'] > 0:
        self.input_ids.append(torch.tensor(encodings_dict['overflowing_tokens'].append(tokenizer.pad_token)) #to desired max length???
        self.attn_masks.append(torch.tensor(encodings_dict["overflowing_tokens"].mask))

      print('inpu ids tensor', self.input_ids)
      print('maks ids tensor', self.attn_masks)

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]