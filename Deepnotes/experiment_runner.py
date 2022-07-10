import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Experiment:
    def __init__(self, model, criterion, optimizer, learning_rate):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
        
    def train(self, train_loader, n_epochs, device):
        for epoch in range(n_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, device)
            print("Epoch %d, train_loss: %.4f, train_acc: %.4f" % (epoch, train_loss, train_acc))
            torch.save(self.model.state_dict(), "./model.pth")
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
    def train_epoch(self, train_loader, device):
        self.model.train()
        train_loss, train_acc = 0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
        
        return train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)
    
    def validate(self, val_loader, device):
        self.model.eval()
        val_loss, val_acc = 0, 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.argmax(1) == labels).sum().item()
            
        return val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset)
        
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
class MLP(nn.Module):
    def __init__(self, in_features, neurons, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.log_softmax(self.fc3(x))
        return out
        
if __name__ == '__main__':
    model = MLP(10, 100, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    experiment = Experiment(model, criterion, optimizer, 0.001)

    train_dataset = torch.tensor(np.random.randn(100, 10), dtype=torch.float32)
    train_labels = torch.tensor(np.random.randint(10, size=(100,)), dtype=torch.int64)
    train_loader = torch.utils.data.DataLoader(list(zip(train_dataset, train_labels)), batch_size=10)

    val_dataset = torch.tensor(np.random.randn(25, 10), dtype=torch.float32)
    val_labels = torch.tensor(np.random.randint(10, size=(25,)), dtype=torch.int64)
    val_loader = torch.utils.data.DataLoader(list(zip(val_dataset, val_labels)), batch_size=5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    early_stopping = EarlyStopping()

    n_epochs = 20
    
    for epoch in range(n_epochs):
        train_loss, train_acc = experiment.train_epoch(train_loader, device)
        print("Epoch %d, train_loss: %.4f, train_acc: %.4f" % (epoch, train_loss, train_acc))
        val_loss, val_acc = experiment.validate(val_loader, device)
        print("Epoch %d, val_loss: %.4f, val_acc: %.4f" % (epoch, val_loss, val_acc))
        adjust_learning_rate(optimizer, 0.01)
        experiment.history["val_loss"].append(val_loss)
        experiment.history["val_acc"].append(val_acc)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Get the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # or load from a file
    # model = MLP(10, 100, 10)
    # model.load_state_dict(torch.load('./model.pth'))

    test_dataset = torch.tensor(np.random.randn(25, 10), dtype=torch.float32)
    test_labels = torch.tensor(np.random.randint(10, size=(25,)), dtype=torch.int64)
    test_loader = torch.utils.data.DataLoader(list(zip(test_dataset, test_labels)), batch_size=5)
    # test_loader = torch.utils.data.DataLoader(some_data_set, batch_size=1)
    test_loss, test_acc = experiment.validate(test_loader, device)
    print("test_loss: %.4f, test_acc: %.4f" % (test_loss, test_acc))

    # to verify the test accuracy is correct
    with torch.no_grad():
        correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 25 test images: %d %%' % (100 * correct / len(test_loader.dataset)))
