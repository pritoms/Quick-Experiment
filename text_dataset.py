import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from bytes_tokenizer import *
import numpy as np

class TextDataset(Dataset):
    def __init__(self, filepath, max_len=100):
        self.filepath = filepath
        self.max_len = max_len
        
        with open(self.filepath, 'r') as f:
            self.data = f.readlines()
            
        self.data = [encode(line, max_len=self.max_len, return_type='tensor') for line in self.data]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx].long()
