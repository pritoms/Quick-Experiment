# `train.py`

from process_handler import Process, ProcessManager
from bytes_tokenizer import encode, decode, tokenizer
from text_dataset import TextDataset
from transformer_model import create_model, create_mask
from checkpoint_manager import *
from train_eval_utils import *
from modeling_utils import *
from Learning Experiment import LearningExperiment
import numpy as np
import time
import torch
import os
import shutil
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Any
from torch.nn import Module


def main():
    # Create an instance of the Model class and define the model parameters. 
    model = LearningExperiment(model=create_model(ntoken=len(tokenizer), 
                                                  ninp=512, 
                                                  nhead=4, 
                                                  nhid=512, 
                                                  nlayers=2, 
                                                  device='cpu'), 
                               model_params={'ntoken': len(tokenizer), 
                                             'ninp': 512, 
                                             'nhead': 4, 
                                             'nhid': 512, 
                                             'nlayers': 2, 
                                             'device': 'cpu'}, 
                               optimizer_function='adam', 
                               criterion_function='cross-entropy', 
                               learning_rate=1e-3, 
                               device='cpu', 
                               checkpoint_dir='checkpoint')

    # Create the data loaders.
    train_dataset = TextDataset(filepath='data/train.txt')
    test_dataset = TextDataset(filepath='data/test.txt')

    # Run the training experiment.
    model.run_experiment(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=32, epochs=3)

if __name__ == '__main__':
    main()