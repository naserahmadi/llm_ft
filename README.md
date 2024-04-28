# LLM Fine Tuning

This repository contains code for fine tuning **llama2** or **llama3** models on two datasets.

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

A btter metric is to use bigger LLMs (e.g. **GPT-4**) or Human to compare the results. 

# Training Parameters
In this section we explain different parameters that we used for training the model. There are two kinds of parameters, fine tuning parameters and PEFT parameters. 

## fine tuning parameters
These parameters are general parameters related to the fine tuning of the model. Here we define parameters such as  ``number of epochs``, ``batch size``, ``accumulation steps``, ``saving strategies``, ``logging``, and ``learning rate``. 

# Selecting Fine tuning Paradigm
