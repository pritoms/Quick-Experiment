from process_handler import Process, ProcessManager
from bytes_tokenizer import encode, decode, tokenizer
from transformer_model import create_model, create_mask
from text_dataset import TextDataset
from checkpoint_manager import *
from train_eval_utils import *
from modeling_utils import *

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

class LearningExperiment:
    def __init__(self, model: Module, 
                 model_params: Dict[str, Any], 
                 optimizer_function: str = 'adam', 
                 criterion_function: str = 'cross-entropy', 
                 learning_rate: float = 1e-3, 
                 device: str = 'cpu', 
                 checkpoint_dir: str = 'checkpoint'):
        self.model = model
        self.model_params = model_params
        self.optimizer_function = optimizer_function
        self.criterion_function = criterion_function
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer: Optimizer = None
        self.criterion: nn.Module = None

    def setup(self):
        self.optimizer = get_optimizer(self.model, self.optimizer_function, self.learning_rate)
        self.criterion = get_criterion(self.criterion_function)

    def load_checkpoint(self, checkpoint: str):
        load_checkpoint(checkpoint, self.model, self.optimizer)

    def save_checkpoint(self, is_best: bool, checkpoint: str):
        state = {
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict()
        }
        save_checkpoint(state, is_best, checkpoint)

    def train(self, train_dataset, batch_size: int = 32, epochs: int = 100):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            avg_loss = train(self.model, train_loader, self.optimizer, self.criterion, self.device)
            print('[Epoch {}] Train loss: {:.4f}'.format(epoch, avg_loss))

    def evaluate(self, test_dataset, batch_size: int = 32):
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        avg_loss = evaluate(self.model, test_loader, self.criterion, self.device)
        print('Test loss: {:.4f}'.format(avg_loss))

    def train_and_evaluate(self, train_dataset: TextDataset, 
                           test_dataset: TextDataset, 
                           batch_size: int = 32, 
                           epochs: int = 100):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            train_avg_loss = train(self.model, train_loader, self.optimizer, self.criterion, self.device)
            test_avg_loss = evaluate(self.model, test_loader, self.criterion, self.device)
            print('[Epoch {}] Train loss: {:.4f} | Test loss: {:.4f}'.format(epoch, train_avg_loss, test_avg_loss))

    def run(self, train_dataset: TextDataset, test_dataset: TextDataset, batch_size: int = 32, epochs: int = 100):
        self.setup()
        self.train_and_evaluate(train_dataset, test_dataset, batch_size=batch_size, epochs=epochs)

        is_best = True
        self.save_checkpoint(is_best, self.checkpoint_dir)

    def run_from_checkpoint(self, train_dataset: TextDataset, test_dataset: TextDataset, checkpoint: str, batch_size: int = 32, epochs: int = 100):
        self.setup()
        self.load_checkpoint(checkpoint)
        self.train_and_evaluate(train_dataset, test_dataset, batch_size=batch_size, epochs=epochs)

        is_best = True
        self.save_checkpoint(is_best, self.checkpoint_dir)

    def run_experiment(self, train_dataset: TextDataset, test_dataset: TextDataset, batch_size: int = 32, epochs: int = 100):
        self.setup()
        if os.path.exists(self.checkpoint_dir):
            self.run_from_checkpoint(train_dataset, test_dataset, checkpoint=os.path.join(self.checkpoint_dir, 'last.pth.tar'), batch_size=batch_size, epochs=epochs)
        else:
            self.run(train_dataset, test_dataset, batch_size=batch_size, epochs=epochs)

    def generate_text(self, prompt: str, max_len: int, temperature: float = 1.0):
        model.eval()
        with torch.no_grad():
            input_ids = encode(prompt, max_len=max_len, return_type='tensor').to(self.device)
            output = model(input_ids)

            output = output[-1, :].unsqueeze(0)
            output = sample_from_distribution(output, temperature)

            input_ids = torch.cat((input_ids, output), dim=0)
            output = decode(input_ids)
        return output

    def run_generator(self, prompt: str, max_len: int, temperature: float = 1.0):
        print(self.generate_text(prompt, max_len, temperature))
