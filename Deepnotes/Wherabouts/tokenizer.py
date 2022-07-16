import torch
import torch.nn as nn
import pickle
import torch
import numpy as np
import os

class BytesTokenizer:
    def __init__(self, num_reserved_tokens=6):
        self.num_reserved_tokens = num_reserved_tokens

    def encode(self, string):
        return np.frombuffer(string, dtype=np.uint8)

    def decode(self, tokens):
        return bytes(tokens)

    def __len__(self):
        return 256 + self.num_reserved_tokens

tokenizer = BytesTokenizer()

def encode(string, max_len=None, return_type='tensor'):
    utf8_bytes = string.encode('utf-8')
    encoded_string = tokenizer.encode(utf8_bytes)
    if max_len is not None:
        if len(encoded_string) > max_len:
            encoded_string = encoded_string[:max_len]
        elif len(encoded_string) < max_len:
            encoded_string = np.pad(encoded_string, (0, max_len - len(encoded_string)), 'constant', constant_values=0)
    if return_type == 'tensor':
        return torch.LongTensor(encoded_string)
    elif return_type == 'numpy':
        return encoded_string
    else:
        raise ValueError('return_type must be either \'tensor\' or \'numpy\'')

def decode(input_ids, return_type='string'):
    if type(input_ids) == torch.LongTensor:
        input_ids = input_ids.numpy()
    decoded_bytes = tokenizer.decode(input_ids)
    if return_type == 'string':
        return decoded_bytes.decode('utf-8')
    elif return_type == 'bytes':
        return decoded_bytes
    else:
        raise ValueError('return_type must be either \'string\' or \'bytes\'')
