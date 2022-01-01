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
X = np.array(X).reshape(-1, T)
Y = np.array(Y).reshape(-1, 1)
N = len(X)

# autoregressive model
# create model
model = nn.Linear(T, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# transform data
X_train = torch.from_numpy(X[: -N // 2].astype(np.float32))
Y_train = torch.from_numpy(Y[: -N // 2].astype(np.float32))
X_test = torch.from_numpy(X[-N // 2:].astype(np.float32))
Y_test = torch.from_numpy(Y[-N // 2:].astype(np.float32))

# train model
n_epochs = 200
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
last_x = torch.from_numpy(X[-N // 2].astype(np.float32))
while len(validation_predictions) < len(validation_target):
    input_ = last_x.view(1, -1)
    p = model(input_)
    validation_predictions.append(p[0, 0].item())
    last_x = torch.cat((last_x[1:], p[0]))

plt.figure()
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
