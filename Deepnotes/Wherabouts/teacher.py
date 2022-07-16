import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from student import Student

class Teacher:
    def __init__(self, dataloader):
        self.memory = {}
        self.dataloader = dataloader
    
    def register_student(self, student, x, y):
        if (x, y) not in self.memory:
            origin = {
                "x": x,
                "y": y
            }
            self.memory[(x, y)] = [student]
        else:
            self.memory[(x, y)].append(student)
    
    def lesson(self):
        for i, data in enumerate(self.dataloader):
            x = data['input']
            y = data['label']
            students = self.memory.get((x, y), None)
            if students is not None:
                for student in students:
                    event = student.step(x)
                    response = {
                        "x": event['recieved'],
                        "y": y,
                        "predicted": event['predicted']
                    }
                    student.confirm(response)
    
    def confirm(self):
        for student in self.memory.values():
            student.train()
