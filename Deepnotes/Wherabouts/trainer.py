import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from teacher import Teacher
from student import Student

class Trainer:
    def __init__(self, dataloader, teacher):
        self.dataloader = dataloader
        self.teacher = teacher
        self.students = []
    
    def build_class(self):
        for i, data in enumerate(self.dataloader):
            x = data['input']
            y = data['label']
            input_size, output_size = len(x), len(y)
            for student in self.students:
                if student.model.input_size >= input_size:
                    if student.model.output_size >= output_size:
                        self.teacher.register_student(student, x, y)
                else:
                    new_student = Student(input_size, input_size + output_size * 2, output_size)
                    self.students.append(new_student)
                    self.teacher.register_student(new_student, x, y)
        
    def evaluate(self):
        for student in self.students:
            print("*" * 80)
            print("student loss", student.loss)
            print("*" * 80)
    
    def train(self):
        for student in self.students:
            student.train()
            student.history = []
            student.loss = 0
