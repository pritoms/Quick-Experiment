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
    """
    Encode a string into a list of integers.

    Args:
        string (str): The string to encode.
        max_len (int, optional): Maximum length of the encoded list.
            If the string is longer than this, it will be truncated.
            If the string is shorter than this, it will be padded with zeros.
            Defaults to None.
        return_type (str, optional): The type of object to return.
            Must be either 'tensor' or 'numpy'. Defaults to 'tensor'.

    Returns:
        encoded_string (torch.Tensor or np.ndarray): The encoded string.
    """
    utf8_bytes = string.encode('utf-8')
    encoded_string = tokenizer.encode(utf8_bytes)
    if max_len is not None:
        if len(encoded_string) > max_len:
            encoded_string = encoded_string[:max_len]
        elif len(encoded_string) < max_len:
            encoded_string = np.pad(encoded_string, (0, max_len - len(encoded_string)), 'constant', constant_values=0)
    if return_type == 'tensor':
        return torch.from_numpy(encoded_string)
    elif return_type == 'numpy':
        return encoded_string
    else:
        raise ValueError('return_type must be either \'tensor\' or \'numpy\'')

def decode(input_ids, return_type='string'):
    """
    Decode a list of integers into a string.

    Args:
        input_ids (torch.Tensor or np.ndarray): The ids corresponding to encoded string.
        return_type (str, optional): The type of object to return.
            Must be either 'string' or 'bytes'. Defaults to 'string'.

    Returns:
        string (str or bytes): The decoded string.
    """
    # Conver the `input_ids` to a numpy array if necessary
    if type(input_ids) == torch.Tensor:
        input_ids = input_ids.numpy()

    # Decode the bytes into a string
    decoded_bytes = tokenizer.decode(input_ids)
    if return_type == 'string':
        return decoded_bytes.decode('utf-8')
    elif return_type == 'bytes':
        return decoded_bytes
    else:
        raise ValueError('return_type must be either \'string\' or \'bytes\'')
