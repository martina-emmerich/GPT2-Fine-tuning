# GPT2-Fine-tuning

Fine-tuning GPT2 model on HuggingFace to change its writing style. Used Jane Asuten dataset from Kaggle: https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens

Model output samples after fine-tuning on the Jane Austen dataset: 


All optional arguments to pass to the script and default values:
{'data': 'austen_dataset.pkl', 'batch_size': 8, 'max_input_length': 100, 'train_size': 0.9, 'max_output_length': 200, 'top_k': 50, 'top_p': 0.95, 'model': 'gpt2', 'warmup_steps': 100.0, 'sample_every': 100, 'epochs': 4, 'lr': 0.0005, 'eps': 1e-08, 'check_dir': 'Checkpoints', 'restore_file': None, 'save_interval': 1, 'load_checkpoint': False, 'output_dir': 'GPT2_fine_tuned_Austen'}

Examples of running the script: python train.py --sample-every 709 --max-input-length 500

To load a checkpoint and continue training run: python train.py --restore-file Epoch_2 --load-checkpoint True