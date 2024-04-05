# GPT2-Fine-tuning

Fine-tuning GPT2 model on HuggingFace to change its writing style. Used Jane Asuten dataset from Kaggle: https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens 

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

Prompt: ***'dancing on the balcony'***

### Gpt2: 
***example 1:*** dancing on the balcony on the way home from work.

"We had gotten our groceries together and it was just like, 'Well, it's time to go outside!' And that's kind of how I saw it when I got up to leave," she said.

The two women also said they never had a "sex scandal" like it did for several women in Chicago in the '60s.

"I mean, they'd be so proud, but like, 'I know what you want for Halloween. I know you need to do something for Halloween,' so you just want to do it."

Devin said that he is "absolutely floored" to hear that some of his neighbors are openly gay. But he also thinks he "still thinks that maybe it's time that the Chicago police do something about it."

"What do you call a city that is like Chicago, like you have a police chief that's like 'Oh, you gotta come out' but you're in this community and all of a sudden you have such a crazy idea about everything you do and what you're doing?" he said. "How can that really happen?"

***example 2:*** dancing on the balcony and the scene was interrupted by the sound of the shots being made.

Mr DeSoto, 27, a student at Chiscot House in Hackney, was in the audience when the shooting occurred.

He was in his home and noticed what he thought was a white van travelling towards him at one end of the venue, heading right up towards him.

The vehicle stopped but the victim was not able to get over to him, with Ms DeSoto shouting, "No more shooting at him."

A passer-by who was standing on the street behind the attacker also heard a shot and said they couldn't believe what was happening

He said he had never seen the police officer so clearly, and that he had no idea where it was coming from.

"We thought the van was coming straight at us from the end of the building, but in fact, I just saw a white van," Mr DeSoto said.

As he and other pedestrians took cover on the ground, one shot was fired, and Mr DeSoto said that while it was the first time he'd seen something like this before, in his home city he had been told it had been coming at him from a long way off.

The victim was later arrested in connection with the shooting.

Mr DeSoto, who is a teacher, said he believes it was only a matter of time before anyone found him.

### Fine-tuned gpt2:
***example 1:***  dancing on the balcony and the house; and though she was too good-humoured to admire Edward, she found herself soon in the midst of his behaviour on hearing this declaration, for though his behaviour to Miss Fairfax, as well as to Mrs. Grant, had been so well concealed before, the whole affair seemed all but forgotten in his coming home. They were sitting near the fire, and she began to suspect the consciousness of a violent passion; and as this was quite evident, in thinking of Edward's meaning it in the same quarter, she made up her mind to what had passed. There was a general feeling of ungraciousness on Edward's side, which did not necessarily follow from the same origin or explanation;
but from his own account she was convinced that he had not. Her friend had asked her a good many questions, and had spoken of Edward's father's opinion of all the evils it could
be supposed to cause, but only by a good-humoured assurance of her own good opinion of Edward and his mother's kindness.

***example 2:***  dancing on the balcony, and leaning against an old bench, when the sound of a gentleman approaching her, or at intervals approaching her, was very distinct, she found herself perfectly equal to the gentleman, who had just passed through the house, and at length approached her.  A few moments were enough for the lady in question, who was still only on the step, who saw the gentlemen entering, to say:

### All optional arguments to pass to the script and default values:
{'data': 'austen_dataset.pkl', 'batch_size': 8, 'max_input_length': 100, 'train_size': 0.9, 'max_output_length': 200, 'top_k': 50, 'top_p': 0.95, 'model': 'gpt2', 'warmup_steps': 100.0, 'sample_every': 100, 'epochs': 4, 'lr': 0.0005, 'eps': 1e-08, 'check_dir': 'Checkpoints', 'restore_file': None, 'save_interval': 1, 'load_checkpoint': False, 'output_dir': 'GPT2_fine_tuned_Austen'}

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

Commamnd line arguments for QLORA parameters will be added shortly.