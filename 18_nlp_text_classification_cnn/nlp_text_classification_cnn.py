import torch
import torch.nn as nn
import torch.nn.functional as F
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


# cnn model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# create model
class CNN(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_outputs):
        super(CNN, self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.K = n_outputs

        self.embed = nn.Embedding(self.V, self.D)

        self.conv1 = nn.Conv1d(self.D, 32, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)

        self.fc = nn.Linear(128, self.K)

    def forward(self, X):
        out = self.embed(X)

        # note: output of embedding is always (N, T, D), but conv1d expects (N, D, T)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)

        out = out.permute(0, 2, 1)
        out, _ = torch.max(out, 1)

        out = self.fc(out)
        return out


model = CNN(len(word2idx), 20, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

train_gen = lambda: data_generator(train_sentences_as_int, df_train.b_labels)
test_gen = lambda: data_generator(test_sentences_as_int, df_test.b_labels)

# train model
n_epochs = 8
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

# make predictions
text = df_test.iloc[1024]['data']
tokens = text.lower().split()
text_int = [word2idx[token] if token in word2idx else 1 for token in tokens]
text_tensor = torch.from_numpy(np.array([text_int])).long()
with torch.no_grad():
    out = model(text_tensor.to(device))
print(text)
print(out)
