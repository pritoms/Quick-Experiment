import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_model import *
from tqdm import tqdm, trange

def train(model, train_loader, optimizer, criterion, device, mask=None):
    model.train()
    total_loss = 0
    for batch_idx, input_ids in enumerate(tqdm(train_loader)):
        input_ids = input_ids.to(device)
        optimizer.zero_grad()
        output = model(input_ids[:, :-1], input_mask=mask)

        loss = criterion(output.view(-1, output.size(-1)), input_ids[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device, mask=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, input_ids in enumerate(tqdm(test_loader)):
            input_ids = input_ids.to(device)
            output = model(input_ids[:, :-1], input_mask=mask)

            loss = criterion(output.view(-1, output.size(-1)), input_ids[:, 1:].contiguous().view(-1))

            total_loss += loss.item()
    return total_loss / len(test_loader)
