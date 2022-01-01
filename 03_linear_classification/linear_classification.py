import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
data = load_breast_cancer()

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# linear classification
# create model
model = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# transform data
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

# train model
n_epochs = 1000
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for it in range(n_epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    output_test = model(X_test)
    loss_test = criterion(output_test, y_test)
    test_losses.append(loss_test.item())

    with torch.no_grad():
        p_train = np.round(output.numpy())
        train_acc.append(np.mean(y_train.numpy() == p_train))

        p_test = np.round(output_test.numpy())
        test_acc.append(np.mean(y_test.numpy() == p_test))

    if (it + 1) % 100 == 0:
        print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')

print(f'Train Acc: {train_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}')

# plot losses
plt.figure('losses')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.savefig('losses.png')

# plot accuracy
plt.figure('accuracy')
plt.plot(train_acc, label='train accuracy')
plt.plot(test_acc, label='test accuracy')
plt.legend()
plt.savefig('accuracy.png')

# save model
torch.save(model.state_dict(), 'linear_classification.pt')

# load saved model
model2 = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
model2.load_state_dict(torch.load('linear_classification.pt'))

# evaluate new model
with torch.no_grad():
    p_train2 = model2(X_train)
    p_train2 = np.round(p_train2.numpy())
    train_acc2 = np.mean(y_train.numpy() == p_train2)

    p_test2 = model2(X_test)
    p_test2 = np.round(p_test2.numpy())
    test_acc2 = np.mean(y_test.numpy() == p_test2)

print(f'New Train Acc: {train_acc2:.4f}, New Test Acc: {test_acc2:.4f}')
