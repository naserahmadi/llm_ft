# LLM Fine Tuning

This repository contains code for fine tuning **llama2** or **llama3** models on two datasets: `samsum` dataset which is a summarization dataset and `trivia-qa` which is a context-base question answering dataset. After downloading and cleaning, we combine two datasets into one training dataset (with different prompt template) and we use this dataset to fine tune a llama model using PEFT. 

After cloning the repo, follow these steps: 
- Install the requirements with: ``pip install -r requirements.txt``
- Download and clean datasets using ``python build_dataset.py``
- run ``train_llama.py`` to start fine tuning llama model. You can change the model from llama2 to llama3 by going in the file. 
- run ``test_llama.py`` to test the output of the model on created test dataset.
- ``test_llama.py`` script reports **preplexity** score on test datasets generates two ``.json`` files containing the results for datasets.
- ``evaluations.py`` reports **ROUGE**, **BLEU** and **EM** by comparing the predictions and references. 
- use ``merge_llama.py`` to merge the created adapter with base model.

Current versions, provide two sets of evaluations metrics: 
- LLM based evaluation metric (**preplexity**)
- Classic IR evaluation metrics (**ROUGE**, **BLUE**, and **EM**)

A state of the art metric is to use bigger LLMs (e.g. **GPT-4**) to compare the generate results with ground truth and return a score between 0 to 1. 

# Training Parameters
In this section we explain different parameters that we used for training the model. There are two kinds of parameters, fine tuning parameters and PEFT parameters. 

## fine tuning parameters
These parameters are general parameters related to the fine tuning of the model. Here we define parameters such as:  
- ``num_train_epochs``: The total number of training epochs
- ``per_device_train_batch_size``: The batch size per GPU for training. This size is limited by the gpus that you have. 
- ``accumulation steps``: The number of gradient accumulation steps before performing a backward pass
- ``saving strategies`` : Define and limit saving checkpoints
- ``logging``: Strategies for logging and reporting
- ``learning_rate``: The initial learning rate for training the model
- ``warmup_steps``: The number of warmup steps for the learning rate scheduler

## PEFT parameters

As mentioned before, we use ``PEFT`` for fine tuning the models. Some of the training parameters are related to ``PEFT``. We load the model in `4bit` format for fine tuning. In PEFT configuration we set the following parameters: 
- `LORA rank (r)` which defines the size of LORA matrices. The bigger this rank, the more parameters we use in fine tuning. It is recommended to use bigger ranks such as 64, but in our case, we used 8.
- `LORA ALPHA` which sets the weight of lora parameters when they are merged with original parameters.
- `TARGET MODULES` which is the most important parameters and gets a list of modules which we want to be updated during the fine tuning. It is recommended to put all linear modules in this list.  


# Selecting Fine tuning Paradigm
We used one model for training both of the tasks instead of training each model. This decision was made for several reasons: 
- Fine-tuning a single model for multiple tasks allows for shared representations across tasks which leads to better generalization and parameter efficiency
- Transfer learning from one task to another can occur more naturally in a unified model which improves the performance on both tasks. Specially in cases which tasks are similar (like our case) the transfer learning can help model to perform better
- With a unified model, deployment and maintenance will be simpler
- Fine-tuning a single model requires less overall training time 
- In this way, the model will adopt to new domains easier 
