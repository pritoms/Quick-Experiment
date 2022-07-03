import torch
import numpy as np
import math
from typing import Callable
from torch.optim import Optimizer
from torch import Tensor
from torch.nn import Module
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss


def get_optimizer(model: Module, optimizer: str = 'adam', learning_rate: float = 1e-3):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

    else:
        raise ValueError('optimizer must be either \'adam\' or \'sgd\'')

def get_criterion(criterion: str = 'cross-entropy'):
    if criterion == 'cross-entropy':
        return CrossEntropyLoss()

    elif criterion == 'mse':
        return MSELoss()

    elif criterion == 'nll':
        return NLLLoss()

    else:
        raise ValueError('criterion must be either \'cross-entropy\', \'mse\', or \'nll\'')

def sample_from_distribution(distribution, temperature=1.0):
    distribution = np.log(distribution) / temperature
    distribution = distribution - np.max(distribution)
    distribution = np.exp(distribution) / np.sum(np.exp(distribution))

    probabilities = torch.from_numpy(distribution).float()
    choices = torch.multinomial(probabilities, 1)
    return choices
