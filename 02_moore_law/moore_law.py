import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('moore.csv', header=None).values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

# process data
Y = np.log(Y)
mx, sx = X.mean(), X.std()
my, sy = Y.mean(), Y.std()
X = (X - mx) / sx
Y = (Y - my) / sy

# linear regression
# create model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

# transform data
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

# train model
n_epochs = 100
losses = []

for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

# plot losses
plt.figure('losses')
plt.plot(losses)
plt.savefig('losses.png')

# predict
predictions = model(inputs).detach().numpy()
plt.figure('results')
plt.scatter(X, Y)
plt.plot(X, predictions, 'r')
plt.savefig('results.png')

# check result
w = model.weight.data.numpy()
a = w[0, 0] * sy / sx
print('Time to double:', np.log(2) / a)
