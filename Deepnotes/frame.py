import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import math

def attend(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SelfAttention(nn.Module):
    def __init__(self, num_features):
        super(SelfAttention, self).__init__()
        self.num_features = num_features
        self.compress = nn.Conv1d(in_channels=num_features, out_channels=num_features//2, kernel_size=1)
        self.query = nn.Conv1d(in_channels=num_features//2, out_channels=num_features//2, kernel_size=1)
        self.key = nn.Conv1d(in_channels=num_features//2, out_channels=num_features//2, kernel_size=1)
        self.value = nn.Conv1d(in_channels=num_features//2, out_channels=num_features//2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        projected_query = self.query(self.compress(x).transpose(1,2)).transpose(1,2)
        projected_key = self.key(self.compress(x).transpose(1,2))
        energy = torch.bmm(projected_query, projected_key)
        attention = self.softmax(energy)
        projected_value = self.value(self.compress(x).transpose(1,2))
        out = torch.bmm(attention, projected_value)
        out = torch.cat((x, out), dim=2)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(num_features=num_features)
        self.conv2 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_features=num_features)
        
    def forward(self, x):
        output = F.relu(self.norm1(self.conv1(x)))
        output = F.relu(self.norm2(self.conv2(output)))
        return x + output
    
class Encoder(nn.Module):
    def __init__(self, num_features=64, hidden_size=128, num_heads=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.query = nn.Linear(num_features, hidden_size)
        self.key = nn.Linear(num_features, hidden_size)
        self.value = nn.Linear(num_features, hidden_size)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        attention, self.attention_map = attend(query, key, value, dropout=self.dropout)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch_size, sequence_length, -1)
        return attention
    
class EncoderBlock(nn.Module):
    def __init__(self, num_features=64, hidden_size=128, num_heads=2, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.query = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.key = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.value = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        attention, self.attention_map = attend(query, key, value, dropout=self.dropout)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch_size, sequence_length, -1)
        return attention
    
class DecoderBlock(nn.Module):
    def __init__(self, num_features=64, hidden_size=128, num_heads=2, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.query = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.key = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.value = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        attention, self.attention_map = attend(query, key, value, dropout=self.dropout)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch_size, sequence_length, -1)
        return attention

class OutputBlock(nn.Module):
    def __init__(self, num_features=64, hidden_size=128, num_heads=2, dropout=0.1):
        super(TransformerOutputBlock, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.query = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.key = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)
        self.value = nn.Conv1d(in_channels=num_features, out_channels=hidden_size, kernel_size=1)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1,2)
        attention, self.attention_map = attend(query, key, value, dropout=self.dropout)
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(batch_size, sequence_length, -1)
        return attention
