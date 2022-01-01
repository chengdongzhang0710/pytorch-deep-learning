import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv('sbux.csv')
df['preClose'] = df['close'].shift(1)
df['return'] = (df['close'] - df['preClose']) / df['preClose']
input_data = df[['open', 'high', 'low', 'close', 'volume']].values
targets = df['return'].values

T = 10
D = input_data.shape[1]
N = len(input_data) - T

Ntrain = len(input_data) * 2 // 3
scaler = StandardScaler()
scaler.fit(input_data[: Ntrain + T - 1])
input_data = scaler.transform(input_data)

X_train = np.zeros((Ntrain, T, D))
Y_train = np.zeros((Ntrain, 1))
for t in range(Ntrain):
    X_train[t, :, :] = input_data[t: t + T]
    Y_train[t] = (targets[t + T] > 0)

X_test = np.zeros((N - Ntrain, T, D))
Y_test = np.zeros((N - Ntrain, 1))
for u in range(N - Ntrain):
    t = u + Ntrain
    X_test[u, :, :] = input_data[t: t + T]
    Y_test[u] = (targets[t + T] > 0)

# rnn model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# create model
class RNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

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

        out, _ = self.rnn(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = RNN(5, 50, 2, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# transform data
X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# train model
n_epochs = 300
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    train_losses[it] = loss.item()

    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, Y_test)
    test_losses[it] = test_loss.item()

    if (it + 1) % 5 == 0:
        print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# plot losses
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# calculate accuracy
with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.cpu().numpy() > 0)
    train_acc = np.mean(Y_train.cpu().numpy() == p_train)

    p_test = model(X_test)
    p_test = (p_test.cpu().numpy() > 0)
    test_acc = np.mean(Y_test.cpu().numpy() == p_test)
print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
