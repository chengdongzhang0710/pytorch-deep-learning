import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# generate data point
N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2 * X[:, 0]) + np.cos(3 * X[:, 1])

# plot data point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# artificial neural networks
# create model
model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# transform data
X_train = torch.from_numpy(X.astype(np.float32)).to(device)
Y_train = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1)).to(device)  # don't forget to reshape!

# train model
n_epochs = 1000
losses = np.zeros(n_epochs)

for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    losses[it] = loss.item()

    if (it + 1) % 100 == 0:
        print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {loss.item():.4f}')

# plot losses
plt.plot(losses)
plt.show()

# plot prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

with torch.no_grad():
    line = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(line, line)
    X_grid = np.vstack((xx.flatten(), yy.flatten())).T
    X_grid_torch = torch.from_numpy(X_grid.astype(np.float32)).to(device)
    Y_hat = model(X_grid_torch).cpu().numpy().flatten()
    ax.plot_trisurf(X_grid[:, 0], X_grid[:, 1], Y_hat, linewidth=0.2, antialiased=True)
    plt.show()

# can the prediction surface extrapolate?
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

with torch.no_grad():
    line = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(line, line)
    X_grid = np.vstack((xx.flatten(), yy.flatten())).T
    X_grid_torch = torch.from_numpy(X_grid.astype(np.float32)).to(device)
    Y_hat = model(X_grid_torch).cpu().numpy().flatten()
    ax.plot_trisurf(X_grid[:, 0], X_grid[:, 1], Y_hat, linewidth=0.2, antialiased=True)
    plt.show()
