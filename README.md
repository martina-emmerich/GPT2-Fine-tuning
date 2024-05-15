# GPT2-Fine-tuning

Fine-tuning GPT-2 model to produce outputs in Jane Austen's writing style. Used Jane Asuten dataset from Kaggle: https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens. The training data was around 11K short paragrpahs and validation was 1K short paragraphs (90%-10% split).

### Goal 

The goal of this project was to write a fine-tuning loop for GPT2 from scratch. The task was chosen beause adapting an LLM to writng style through fine-tuning does not require huge amounts of training data to see a difference in the model output. Chat models such as Llama2-7B-Chat or more recent GPT chat models can be prompted to obtain similar changes in their writing style while also obtaining better fluency as the base LLMs perform better overall compared to GPT2. Some simple prompting examples with Llama 2-7B-Chat can be seen in the jupyter notebook ```Llama2-7B-chat_prompting.ipynb```.

### Notes on results
Prompting LLama 2 chat gives more fluent and coherent sentences than both fine-tuned GPT2 models, but the style overall seems to be less similar to Jane Austen's writing style than when we fine-tuned GPT-2/GPT2-medium. However, the fine-tuned versions of GPT2 seem more likely to use names directly from the Austen novels they were trained on in their responses compared to the Llama 2 model which might indicate that the GPT2 models are slighly overfitting to the training data. Also different generation parameters and prompts can give better or worse results, with a base LLM like GPT2 the model output is only affected by generation parameters as it is not made to receive user instructions.


## Inference settings and outputs
Generation settings used for both models: 
                                ```
                                model.generate( 
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300, #300
                                top_p=0.95, #0.95 
                                num_return_sequences=3)
                                 ```

Example outputs for each model, fine-tuned and not, can be found in the sample_outputs folder.

## Installation 
At least a 16GB GPU is recommended especially for fine-tuning bigger models like GPT-2 medium.

``` pip install -r requirements.txt ```

## Running fine-tuning 
To run the basic script with the default options (fine-tuning GPT-2 without using QLoRA) use the following command replacing the data field with your data filename:

 ``` python train.py --data 'austen_dataset.pkl ```

To load a checkpoint and continue training run: 

``` python train.py --restore-file Epoch_2 --load-checkpoint True ```

To fine-tune a model (in this case GPT-2 medium) using QLoRA:

``` python train.py --sample-every 709 --max-input-length 500 --qlora True --model 'gpt2-medium' ```

LoRA parameters can be changed using the following flags:

 ``` --rank_lora int, --alpha_lora int , --targets_lora 'c_attn' 'c_proj' ```

 ### All optional arguments to pass to the train.py script and default values:
 ```
 {'data': 'austen_dataset.pkl', 'batch_size': 8, 'max_input_length': 100, 'train_size': 0.9, 'max_output_length': 200, 'top_k': 50, 'top_p': 0.95, 'model': 'gpt2', 'qlora': False, 'rank_lora': 64, 'alpha_lora': 16, 'targets_lora': ['c_attn'], 'warmup_steps': 100.0, 'sample_every': 100, 'epochs': 4, 'lr': 0.0005, 'eps': 1e-08, 'check_dir': 'Checkpoints', 'restore_file': None, 'save_interval': 1, 'load_checkpoint': False, 'output_dir', 'GPT2_fine_tuned_Austen'}
 
  ```

 ## Running inference

 To run inference on a fine-tuned model the following command can be used:

  ``` python inference.py --model_path 'path_to_saved_model' --prompt 'prompt'  ``` 

 If the model was fine-tuned using LoRA then the following command can be used:

  ``` python inference.py --model_path 'path_to_saved_model' --qlora True --prompt 'prompt' --base_model 'name of the base model on hf'  ```  