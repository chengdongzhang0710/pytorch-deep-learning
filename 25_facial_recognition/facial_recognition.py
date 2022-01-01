import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from glob import glob
from collections import Counter

files = glob('yalefaces/subject*')
np.random.shuffle(files)
N = len(files)
H, W = 60, 80


def load_img(filepath):
    # load image and downsample
    # img = torchvision.transforms.functional.to_tensor(PIL.Image.open(filepath).resize((W, H)))
    # img = torch.from_numpy(np.asarray(PIL.Image.open(filepath).resize((W, H)))).float() / 255
    img = np.asarray(PIL.Image.open(filepath).resize((W, H))) / 255
    return img


# img = load_img(np.random.choice(files))
# print(img.shape)  # -> (60, 80)

# load images as arrays
shape = (N, H, W)
images = np.zeros(shape)
for i, f in enumerate(files):
    image = load_img(f)
    images[i] = image

# make labels
labels = np.zeros(N)
for i, f in enumerate(files):
    filename = f.rsplit('\\', 1)[-1]
    subject_num = filename.split('.', 1)[0]
    idx = int(subject_num.replace('subject', '')) - 1
    labels[i] = idx

label_count = Counter(labels)
unique_labels = set(label_count.keys())
n_subject = len(label_count)
n_test = 3 * n_subject
n_train = N - n_test

train_images = np.zeros((n_train, H, W))
train_labels = np.zeros(n_train)
test_images = np.zeros((n_test, H, W))
test_labels = np.zeros(n_test)

count_so_far = {}
train_idx = 0
test_idx = 0
images, labels = shuffle(images, labels)
for image, label in zip(images, labels):
    count_so_far[label] = count_so_far.get(label, 0) + 1
    if count_so_far[label] > 3:
        train_images[train_idx] = image
        train_labels[train_idx] = label
        train_idx += 1
    else:
        test_images[test_idx] = image
        test_labels[test_idx] = label
        test_idx += 1

train_label2idx = {}
test_label2idx = {}
for i, label in enumerate(train_labels):
    if label not in train_label2idx:
        train_label2idx[label] = [i]
    else:
        train_label2idx[label].append(i)
for i, label in enumerate(test_labels):
    if label not in test_label2idx:
        test_label2idx[label] = [i]
    else:
        test_label2idx[label].append(i)

train_positives = []
train_negatives = []
test_positives = []
test_negatives = []
for label, indices in train_label2idx.items():
    other_indices = set(range(n_train)) - set(indices)
    for i, idx1 in enumerate(indices):
        for idx2 in indices[i + 1:]:
            train_positives.append((idx1, idx2))
        for idx2 in other_indices:
            train_negatives.append((idx1, idx2))
for label, indices in test_label2idx.items():
    other_indices = set(range(n_test)) - set(indices)
    for i, idx1 in enumerate(indices):
        for idx2 in indices[i + 1:]:
            test_positives.append((idx1, idx2))
        for idx2 in other_indices:
            test_negatives.append((idx1, idx2))

# create data generator
batch_size = 64


def train_generator():
    n_batches = int(np.ceil(len(train_positives) / batch_size))
    while True:
        np.random.shuffle(train_positives)
        n_samples = batch_size * 2
        shape = (n_samples, H, W)
        x_batch1 = np.zeros(shape)
        x_batch2 = np.zeros(shape)
        y_batch = np.zeros(n_samples)
        for i in range(n_batches):
            pos_batch_indices = train_positives[i * batch_size: (i + 1) * batch_size]
            j = 0
            for idx1, idx2 in pos_batch_indices:
                x_batch1[j] = train_images[idx1]
                x_batch2[j] = train_images[idx2]
                y_batch[j] = 1  # 1 means match
                j += 1
            neg_indices = np.random.choice(len(train_negatives), size=len(pos_batch_indices), replace=False)
            for neg in neg_indices:
                idx1, idx2 = train_negatives[neg]
                x_batch1[j] = train_images[idx1]
                x_batch2[j] = train_images[idx2]
                y_batch[j] = 0  # 0 means non-match
                j += 1
            x1 = x_batch1[: j]
            x2 = x_batch2[: j]
            y = y_batch[: j]

            x1 = x1.reshape(-1, 1, H, W)
            x2 = x2.reshape(-1, 1, H, W)
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            y = torch.from_numpy(y).float()
            yield [x1, x2], y


def test_generator():
    n_batches = int(np.ceil(len(test_positives) / batch_size))
    while True:
        n_samples = batch_size * 2
        shape = (n_samples, H, W)
        x_batch1 = np.zeros(shape)
        x_batch2 = np.zeros(shape)
        y_batch = np.zeros(n_samples)
        for i in range(n_batches):
            pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]
            j = 0
            for idx1, idx2 in pos_batch_indices:
                x_batch1[j] = test_images[idx1]
                x_batch2[j] = test_images[idx2]
                y_batch[j] = 1  # 1 means match
                j += 1
            neg_indices = np.random.choice(len(test_negatives), size=len(pos_batch_indices), replace=False)
            for neg in neg_indices:
                idx1, idx2 = test_negatives[neg]
                x_batch1[j] = test_images[idx1]
                x_batch2[j] = test_images[idx2]
                y_batch[j] = 0  # 0 means non-match
                j += 1
            x1 = x_batch1[: j]
            x2 = x_batch2[: j]
            y = y_batch[: j]

            x1 = x1.reshape(-1, 1, H, W)
            x2 = x2.reshape(-1, 1, H, W)
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            y = torch.from_numpy(y).float()
            yield [x1, x2], y


# create model
class SiameseNN(nn.Module):
    def __init__(self, feature_dim):
        super(SiameseNN, self).__init__()

        # define cnn featurizer
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(13 * 18 * 64, 128),  # 60 -> 29 -> 13, 80 -> 39 -> 18
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, im1, im2):
        feat1 = self.cnn(im1)
        feat2 = self.cnn(im2)
        return torch.norm(feat1 - feat2, dim=-1)  # euclidean distance between feature 1 and feature 2


model = SiameseNN(50)

# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


def contrastive_loss(y, t):
    non_match = F.relu(1 - y)  # max(margin - y, 0)
    return torch.mean(t * y ** 2 + (1 - t) * non_match ** 2)


optimizer = torch.optim.Adam(model.parameters())


# train model
# batch gradient descent
def batch_gd(model, criterion, optimizer, train_gen, test_gen, train_steps_per_epoch, test_steps_per_epoch, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        steps = 0
        for (x1, x2), targets in train_gen:
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            steps += 1
            if steps >= train_steps_per_epoch:
                break
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []
        steps = 0
        for (x1, x2), targets in test_gen:
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            steps += 1
            if steps >= test_steps_per_epoch:
                break
        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f'Epoch: {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}')

    return train_losses, test_losses


train_steps = int(np.ceil(len(train_positives) / batch_size))
test_steps = int(np.ceil(len(test_positives) / batch_size))
train_losses, test_losses = batch_gd(model, contrastive_loss, optimizer, train_generator(), test_generator(),
                                     train_steps, test_steps, 20)

# plot loss
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# calculate accuracy
model.eval()


def predict(x1, x2):
    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)
    with torch.no_grad():
        dist = model(x1, x2).cpu().numpy()
        return dist.flatten()


def get_train_accuracy(threshold=0.85):
    positive_distances = []
    negative_distances = []
    tp, tn, fp, fn = 0, 0, 0, 0

    batch_size = 64
    x_batch1 = np.zeros((batch_size, 1, H, W))
    x_batch2 = np.zeros((batch_size, 1, H, W))

    n_batches = int(np.ceil(len(train_positives) / batch_size))
    for i in range(n_batches):
        pos_batch_indices = train_positives[i * batch_size: (i + 1) * batch_size]
        j = 0
        for idx1, idx2 in pos_batch_indices:
            x_batch1[j, 0] = train_images[idx1]
            x_batch2[j, 0] = train_images[idx2]
            j += 1
        x1 = x_batch1[: j]
        x2 = x_batch2[: j]
        distances = predict(x1, x2)
        positive_distances += distances.tolist()
        tp += (distances < threshold).sum()
        fn += (distances > threshold).sum()

    n_batches = int(np.ceil(len(train_negatives) / batch_size))
    for i in range(n_batches):
        neg_batch_indices = train_negatives[i * batch_size: (i + 1) * batch_size]
        j = 0
        for idx1, idx2 in neg_batch_indices:
            x_batch1[j, 0] = train_images[idx1]
            x_batch2[j, 0] = train_images[idx2]
            j += 1
        x1 = x_batch1[: j]
        x2 = x_batch2[: j]
        distances = predict(x1, x2)
        negative_distances += distances.tolist()
        fp += (distances < threshold).sum()
        tn += (distances > threshold).sum()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'Sensitivity(tpr): {tpr}, Specificity(tnr): {tnr}')

    plt.figure()
    plt.hist(negative_distances, bins=20, density=True, label='negative distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive distances')
    plt.legend()
    plt.show()


def get_test_accuracy(threshold=0.85):
    positive_distances = []
    negative_distances = []
    tp, tn, fp, fn = 0, 0, 0, 0

    batch_size = 64
    x_batch1 = np.zeros((batch_size, 1, H, W))
    x_batch2 = np.zeros((batch_size, 1, H, W))

    n_batches = int(np.ceil(len(test_positives) / batch_size))
    for i in range(n_batches):
        pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]
        j = 0
        for idx1, idx2 in pos_batch_indices:
            x_batch1[j, 0] = test_images[idx1]
            x_batch2[j, 0] = test_images[idx2]
            j += 1
        x1 = x_batch1[: j]
        x2 = x_batch2[: j]
        distances = predict(x1, x2)
        positive_distances += distances.tolist()
        tp += (distances < threshold).sum()
        fn += (distances > threshold).sum()

    n_batches = int(np.ceil(len(test_negatives) / batch_size))
    for i in range(n_batches):
        neg_batch_indices = test_negatives[i * batch_size: (i + 1) * batch_size]
        j = 0
        for idx1, idx2 in neg_batch_indices:
            x_batch1[j, 0] = test_images[idx1]
            x_batch2[j, 0] = test_images[idx2]
            j += 1
        x1 = x_batch1[: j]
        x2 = x_batch2[: j]
        distances = predict(x1, x2)
        negative_distances += distances.tolist()
        fp += (distances < threshold).sum()
        tn += (distances > threshold).sum()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f'Sensitivity(tpr): {tpr}, Specificity(tnr): {tnr}')

    plt.figure()
    plt.hist(negative_distances, bins=20, density=True, label='negative distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive distances')
    plt.legend()
    plt.show()


print('Get Train Accuracy')
get_train_accuracy(0.65)

print('Get Test Accuracy')
get_test_accuracy(0.65)
