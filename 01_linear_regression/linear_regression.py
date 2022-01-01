import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# generate data points
N = 20
X = np.random.random(N) * 10 - 5
Y = 0.5 * X - 1 + np.random.randn(N)

# linear regression
# create model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# transform data
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

# train model
n_epochs = 30
losses = []

for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    # print(f'Epoch: {it + 1}/{n_epochs}, Loss: {loss.item():.4f}')

# plot losses
plt.figure('losses')
plt.plot(losses)
plt.savefig('losses.png')

# predict
predictions = model(inputs).detach().numpy()
plt.figure('results')
plt.scatter(X, Y)
plt.plot(X, predictions)
plt.savefig('results.png')

# parameters
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print('Slope:', w)
print('Intercept:', b)
