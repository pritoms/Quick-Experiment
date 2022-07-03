import torch
from LearningExperiment import LearningExperiment
from bytes_tokenizer import tokenizer
from text_dataset import TextDataset
from transformer_model import create_model, create_mask

device = torch.device('cpu')

ntoken = len(tokenizer)
ninp = 512  # embedding dimension
nhead = 8  # the number of heads in the multiheadattention models
nhid = ninp * 4  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
dropout = 0.5  # the dropout value
model_params = {'ntoken': ntoken, 'ninp': ninp, 'nhead': nhead, 'nhid': nhid, 'nlayers': nlayers, 'dropout': dropout}
model = create_model(**model_params).to(device)

optimizer_function = 'adam'
criterion_function = 'cross-entropy'
learning_rate = 1e-3
checkpoint_dir = 'checkpoint'
exp = LearningExperiment(model, model_params, optimizer_function, criterion_function, learning_rate, device, checkpoint_dir)

train_dataset = TextDataset('data/train.txt')
test_dataset = TextDataset('data/test.txt')
exp.run(train_dataset, test_dataset)