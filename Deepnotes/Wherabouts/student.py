
import torch
import torch.nn as nn
from mlp import MLP

class Student:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = MLP(input_size, hidden_size, output_size)
        self.history = []
        self.loss = 0
    
    def step(self, inputs):
        event = {
            "recieved": inputs,
            "predicted": self.model(inputs),
            "learned": None
        }
        return event
    
    def confirm(self, response):
        self.history.append(response)
        loss = self.calculate_loss(response['predicted'], response['learned'])
        self.loss += loss
    
    def calculate_loss(self, predicted, learned):
        return torch.sum(torch.pow(predicted - learned, 2))
    
    def train(self, optimizer=torch.optim.SGD, lr=0):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        for event in self.history:
            x = event['recieved']
            y = event['learned']
            z = self.step(x)
            loss = self.calculate_loss(z['predicted'], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
