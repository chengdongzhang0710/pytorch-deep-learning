import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# generate data
N = 1000
series = np.sin(0.1 * np.arange(N)) + np.random.randn(N) * 0.1
plt.figure()
plt.plot(series)
plt.show()

# build dataset
# use T past values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series) - T):
    x = series[t: t + T]
    X.append(x)
    y = series[t + T]
    Y.append(y)
X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)
N = len(X)

# simple rnn model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# create model
class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        # note: batch_first = True
        # applies the convention that our data will be of shape:
        # (num_samples, sequence_length, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            nonlinearity='relu',
            batch_first=True,
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        # initial hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # out is of size (N, T, M)
        # 2nd return value is hidden states at each hidden layer
        out, _ = self.rnn(X, h0)
        out = self.fc(out[:, -1, :])
        return out


model = SimpleRNN(n_inputs=1, n_hidden=15, n_rnnlayers=1, n_outputs=1)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# transform data
X_train = torch.from_numpy(X[: -N // 2].astype(np.float32))
Y_train = torch.from_numpy(Y[: -N // 2].astype(np.float32))
X_test = torch.from_numpy(X[-N // 2:].astype(np.float32))
Y_test = torch.from_numpy(Y[-N // 2:].astype(np.float32))

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# train model
n_epochs = 1000
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

# predict future values
validation_target = Y[-N // 2:]
validation_predictions = []
last_x = X_test[0].view(T)
while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1, T, 1)
    p = model(input_)
    validation_predictions.append(p[0, 0].item())
    last_x = torch.cat((last_x[1:], p[0]))

plt.figure()
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
