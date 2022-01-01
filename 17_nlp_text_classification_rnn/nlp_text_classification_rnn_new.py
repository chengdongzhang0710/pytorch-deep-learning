import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# load data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
df_train, df_test = train_test_split(df, test_size=0.33)

# text preprocessing
idx = 2
word2idx = {'<PAD>': 0, '<UNKNOWN>': 1}

for i, row in df_train.iterrows():
    tokens = row['data'].lower().split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

train_sentences_as_int = []
for i, row in df_train.iterrows():
    tokens = row['data'].lower().split()
    sentence_as_int = [word2idx[token] for token in tokens]
    train_sentences_as_int.append(sentence_as_int)

test_sentences_as_int = []
for i, row in df_test.iterrows():
    tokens = row['data'].lower().split()
    sentence_as_int = [word2idx[token] if token in word2idx else 1 for token in tokens]
    test_sentences_as_int.append(sentence_as_int)


# transform data
def data_generator(X, Y, batch_size=32):
    X, Y = shuffle(X, Y)
    n_batches = int(np.ceil(len(Y) / batch_size))
    for i in range(n_batches):
        end = min((i + 1) * batch_size, len(Y))
        X_batch = X[i * batch_size: end]
        Y_batch = Y[i * batch_size: end]

        max_len = np.max([len(x) for x in X_batch])
        for j in range(len(X_batch)):
            x = X_batch[j]
            pad = [0] * (max_len - len(x))
            X_batch[j] = pad + x

        X_batch = torch.from_numpy(np.array(X_batch)).long()
        Y_batch = torch.from_numpy(np.array(Y_batch)).long()
        yield X_batch, Y_batch


# rnn model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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


model = RNN(len(word2idx), 20, 15, 1, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

train_gen = lambda: data_generator(train_sentences_as_int, df_train.b_labels)
test_gen = lambda: data_generator(test_sentences_as_int, df_test.b_labels)

# train model
n_epochs = 15
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
for it in range(n_epochs):
    t0 = datetime.now()
    train_loss = []
    for inputs, targets in train_gen():
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.view(-1, 1).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    train_losses[it] = np.mean(train_loss)

    test_loss = []
    for inputs, targets in test_gen():
        inputs, targets = inputs.to(device), targets.to(device)
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
for inputs, targets in train_gen():
    inputs, targets = inputs.to(device), targets.to(device)
    targets = targets.view(-1, 1).float()
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
train_acc = n_correct / n_total

n_correct = 0
n_total = 0
for inputs, targets in test_gen():
    inputs, targets = inputs.to(device), targets.to(device)
    targets = targets.view(-1, 1).float()
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
test_acc = n_correct / n_total

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
