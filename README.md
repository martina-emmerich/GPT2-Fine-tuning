# GPT2-Fine-tuning

Fine-tuning GPT2 model on HuggingFace to change its writing style. Used Jane Asuten dataset from Kaggle: link. 
Model outputs samples in the style of writing on Jane Austen's writing after fine-tuning. 

Examples of running the script:

To load a checkpoint and continue training run:
python train.py --restore-file Epoch_2 --load-checkpoint True