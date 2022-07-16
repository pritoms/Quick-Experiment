import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from student import Student

class Teacher:
    def __init__(self, dataloader, class_size):
        self.dataloader = dataloader
        self.class_size = class_size
    
    def step(self, students):
        for i, data in enumerate(self.dataloader):
            x = data['input']
            y = data['label']
            events = []
            for student in students:
                events.append(student.step(x))

            for j, event in enumerate(events):
                student = students[j]
                event['learned'] = y
                student.confirm(event)
            if (i+1) % self.class_size == 0:
                print(f"Class {i+1} completed")
                for student in students:
                    print(student.loss)
                    student.loss = 0
                print()
                for student in students:
                    student.train()
