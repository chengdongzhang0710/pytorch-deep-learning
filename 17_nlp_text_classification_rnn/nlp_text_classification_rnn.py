import torch
import torch.nn as nn
import torchtext.data as ttd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# load data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
df2 = df[['data', 'b_labels']]
df2.to_csv('spam2.csv', index=False)

TEXT = ttd.Field(
    sequential=True,
    batch_first=True,
    lower=False,
    # tokenize='spacy',
    pad_first=True,
)
LABEL = ttd.Field(
    sequential=False,
    use_vocab=False,
    is_target=True,
)
dataset = ttd.TabularDataset(
    path='spam2.csv',
    format='csv',
    skip_header=True,
    fields=[('data', TEXT), ('label', LABEL)],
)

train_dataset, test_dataset = dataset.split()

TEXT.build_vocab(train_dataset, )
vocab = TEXT.vocab

# rnn model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# transform data
train_iter, test_iter = ttd.Iterator.splits(
    datasets=(train_dataset, test_dataset),
    sort_key=lambda x: len(x.data),
    batch_sizes=(32, 256),
    device=device,
)


# create model
class RNN(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        self.embed = nn.Embedding(self.V, self.D)
        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True,
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        out = self.embed(X)
        out, _ = self.rnn(out, (h0, c0))
        out, _ = torch.max(out, 1)
        out = self.fc(out)
        return out


model = RNN(len(vocab), 20, 15, 1, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# train model
n_epochs = 15
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
for it in range(n_epochs):
    t0 = datetime.now()
    train_loss = []
    for inputs, targets in train_iter:
        targets = targets.view(-1, 1).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    train_losses[it] = np.mean(train_loss)

    test_loss = []
    for inputs, targets in test_iter:
        targets = targets.view(-1, 1).float()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss.append(loss.item())
    test_losses[it] = np.mean(test_loss)

    dt = datetime.now() - t0
    print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {train_losses[it]:.4f}, Test Loss: {test_losses[it]:.4f}, '
          f'Duration: {dt}')

# plot losses
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# calculate accuracy
n_correct = 0
n_total = 0
for inputs, targets in train_iter:
    targets = targets.view(-1, 1).float()
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
train_acc = n_correct / n_total

n_correct = 0
n_total = 0
for inputs, targets in test_iter:
    targets = targets.view(-1, 1).float()
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
test_acc = n_correct / n_total

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
