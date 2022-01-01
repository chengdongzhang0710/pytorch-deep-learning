import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# generate data from a known distribution
def generate_batch(batch_size=32):
    # x in (-5, +5)
    x = np.random.random(batch_size) * 10 - 5

    # standard deviation is a function of x
    sd = 0.05 + 0.1 * (x + 5)

    # target = mean + noise * sd
    y = np.cos(x) - 0.3 * x + np.random.randn(batch_size) * sd

    return x, y


plt.figure()
x, y = generate_batch(1024)
plt.scatter(x, y, alpha=0.5)
plt.show()


# create model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.ann1 = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )
        self.ann2 = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, inputs):
        # returns (mean, log-variance)
        return self.ann1(inputs), self.ann2(inputs)


model = Model()


def criterion(outputs, targets):
    mu = outputs[0]
    v = torch.exp(outputs[1])

    # coefficient term
    c = torch.log(torch.sqrt(2 * np.pi * v))

    # exponent term
    f = 0.5 / v * (targets - mu) ** 2

    # negative mean log-likelihood
    nll = torch.mean(c + f)

    return nll


optimizer = torch.optim.Adam(model.parameters())

n_epochs = 5000
batch_size = 128
losses = np.zeros(n_epochs)
for i in range(n_epochs):
    x, y = generate_batch(batch_size)
    inputs = torch.from_numpy(x).float()
    targets = torch.from_numpy(y).float()
    inputs, targets = inputs.view(-1, 1), targets.view(-1, 1)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses[i] = loss.item()
    loss.backward()
    optimizer.step()

    if (i + 1) % 1000 == 0:
        print(f'Epoch: {i + 1}, Loss: {losses[i]:.4f}')

# plot loss
plt.figure()
plt.plot(losses)
plt.show()

# plot model predictions
x, y = generate_batch(1024)
plt.figure()
plt.scatter(x, y, alpha=0.5)

inputs = torch.from_numpy(x).float()
targets = torch.from_numpy(y).float()
inputs, targets = inputs.view(-1, 1), targets.view(-1, 1)

with torch.no_grad():
    outputs = model(inputs)
    y_hat = outputs[0].numpy().flatten()
    sd = np.exp(outputs[1].numpy().flatten() / 2)

idx = np.argsort(x)
plt.plot(x[idx], y_hat[idx], linewidth=3, color='red')
plt.fill_between(x[idx], y_hat[idx] - sd[idx], y_hat[idx] + sd[idx], color='red', alpha=0.3)
plt.show()
