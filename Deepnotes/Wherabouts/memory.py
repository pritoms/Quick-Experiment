import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Memory(nn.Module):
    def __init__(self, num_slots, slot_size, batch_size):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.batch_size = batch_size
        self.memory = nn.Parameter(torch.randn(num_slots, slot_size))
    
    def forward(self, key, value):
        key_weights = torch.matmul(self.memory, key)
        key_weights = F.softmax(key_weights, dim=0)
        memory_out = torch.matmul(key_weights, value)
        return memory_out
