from experiment_handler import *
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_params = {'d_model': 128, 'nhead': 4, 'd_hid': 256, 'nlayers': 2, 'max_len': 100}
run_experiment(model_params)
