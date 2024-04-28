# LLM Fine Tuning

This repository contains code for fine tuning **LLama2**** or **LLama3** models on two datasets.

After cloning the repo, follow these steps: 
- Install the requirements with: ``pip install -r requirements.txt``
- Download and clean datasets using ``python build_dataset.py``
- run ``train_llama.py`` to start fine tuning llama model. You can change the model from llama2 to llama3 by going in the file. 
- run ``test_llama.py`` to test the output of the model on created test dataset.
- use ``merge_llama.py`` to merge the created adapter with base model.

