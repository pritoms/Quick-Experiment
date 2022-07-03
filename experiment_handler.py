from experiment_manager import LearningExperiment
from bytes_tokenizer import *
from transformer_model import create_model, create_mask
from text_dataset import TextDataset

import torch
import os

def run_experiment(model_params):
    # Define the model parameters
    ntoken = len(tokenizer)
    ninp = model_params['d_model']
    nhead = model_params['nhead']
    nhid = model_params['d_hid']
    nlayers = model_params['nlayers']

    # Create a new model instance
    model = create_model(ntoken, ninp, nhead, nhid, nlayers, device=device)

    # Create a new learning experiment instance
    experiment = LearningExperiment(model, model_params, 
                                    optimizer_function='adam', 
                                    criterion_function='cross-entropy', 
                                    learning_rate=1e-3, 
                                    device=device, 
                                    checkpoint_dir='checkpoint')

    # Load the train and test datasets
    train_dataset = TextDataset('train.txt', max_len=model_params['max_len'])
    test_dataset = TextDataset('test.txt', max_len=model_params['max_len'])

    # Run the experiment!
    experiment.run_experiment(train_dataset, test_dataset, batch_size=32, epochs=100)
