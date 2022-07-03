import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_optimizer(model, function='adam', learning_rate=1e-3):
    if function == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif function == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return optimizer

def get_criterion(function='cross-entropy'):
    if function == 'cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif function == 'mse':
        criterion = torch.nn.MSELoss()
    return criterion