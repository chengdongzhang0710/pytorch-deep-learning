import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# load data and build data loader
transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
test_dataset = datasets.ImageFolder('data/test', transform=transform)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# transfer learning without data augmentation
vgg = models.vgg16(pretrained=True)


class VGGFeatures(nn.Module):
    def __init__(self, vgg):
        super(VGGFeatures, self).__init__()
        self.vgg = vgg

    def forward(self, X):
        out = self.vgg.features(X)
        out = self.vgg.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


Ntrain = len(train_dataset)
Ntest = len(test_dataset)
D = vgg.classifier[0].in_features

X_train = np.zeros((Ntrain, D))
Y_train = np.zeros((Ntrain, 1))
X_test = np.zeros((Ntest, D))
Y_test = np.zeros((Ntest, 1))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
vggf = VGGFeatures(vgg)
vggf.to(device)

# pre populate X_train and Y_train, X_test and Y_test
i = 0
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        outputs = vggf(inputs)
        size = len(outputs)
        X_train[i: i + size] = outputs.cpu().detach().numpy()
        Y_train[i: i + size] = targets.view(-1, 1).numpy()
        i += size

i = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = vggf(inputs)
        size = len(outputs)
        X_test[i: i + size] = outputs.cpu().detach().numpy()
        Y_test[i: i + size] = targets.view(-1, 1).numpy()
        i += size

# logistic regression
scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.fit_transform(X_test)

model = nn.Linear(D, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

train_dataset2 = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train2.astype(np.float32)),
    torch.from_numpy(Y_train.astype(np.float32)),
)
test_dataset2 = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test2.astype(np.float32)),
    torch.from_numpy(Y_test.astype(np.float32)),
)

batch_size = 128
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=batch_size, shuffle=False)


def batch_gd(model, criterion, optimizer, train_loader, test_loader, n_epochs):
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        t0 = datetime.now()

        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}')

    return train_losses, test_losses


train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader2, test_loader2, n_epochs=10)

# plot loss
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# calculate accuracy
n_correct = 0
n_total = 0
for inputs, targets in train_loader2:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
train_acc = n_correct / n_total

n_correct = 0
n_total = 0
for inputs, targets in test_loader2:
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    predictions = (outputs > 0)
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]
test_acc = n_correct / n_total

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
