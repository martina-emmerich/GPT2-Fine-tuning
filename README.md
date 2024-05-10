# GPT2-Fine-tuning

Fine-tuning GPT-2 model to produce outputs in Jane Austen's writing style. Used Jane Asuten dataset from Kaggle: https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens. The training data was around 11K short paragrpahs and validation was 1K short paragraphs (90%-10% split).

Model output samples after fine-tuning on the Jane Austen dataset:
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

### All optional arguments to pass to the script and default values:
 {'data': 'austen_dataset_long.pkl', 'batch_size': 8, 'max_input_length': 500, 'train_size': 0.9, 'max_output_length': 200, 'top_k': 50, 'top_p': 0.95, 'model': 'gpt2-medium', 'qlora': True, 'rank_lora': 64, 'alpha_lora': 16, 'targets_lora': ['c_attn'], 'warmup_steps': 100.0, 'sample_every': 709, 'epochs': 4, 'lr': 0.0005, 'eps': 1e-08, 'check_dir': 'Checkpoints', 'restore_file': None, 'save_interval': 1, 'load_checkpoint': False, 'output_dir': 'GPT2_fine_tuned_Austen'}

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