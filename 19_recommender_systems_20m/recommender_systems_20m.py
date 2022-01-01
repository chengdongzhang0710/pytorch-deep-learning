import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle

# load data
df = pd.read_csv('ml-20m/ratings.csv')

# make the userId and movieId be numbered 0, 1, ..., N-1
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values - 2.5

N = len(set(user_ids))  # number of users
M = len(set(movie_ids))  # number of movies
D = 10  # embedding dimension

# ann model
# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# create model
class Model(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_hidden=1024):
        super(Model, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim

        self.u_emb = nn.Embedding(self.N, self.D)
        self.m_emb = nn.Embedding(self.M, self.D)
        self.fc1 = nn.Linear(2 * self.D, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # set the weights since N(0, 1) leads to poor results
        self.u_emb.weight.data = nn.Parameter(torch.Tensor(np.random.randn(self.N, self.D) * 0.01))
        self.m_emb.weight.data = nn.Parameter(torch.Tensor(np.random.randn(self.M, self.D) * 0.01))

    def forward(self, u, m):
        u = self.u_emb(u)  # output is (num_samples, D)
        m = self.m_emb(m)  # output is (num_samples, D)

        out = torch.cat((u, m), 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


model = Model(N, M, D)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)


# customized batch gradient descent
def batch_gd2(model, criterion, optimizer, train_data, test_data, n_epochs, batch_size=512):
    train_users, train_movies, train_ratings = train_data
    test_users, test_movies, test_ratings = test_data

    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    Ntrain = len(train_users)
    batches_per_epoch = int(np.ceil(Ntrain / batch_size))

    for it in range(n_epochs):
        t0 = datetime.now()
        train_loss = []

        train_users, train_movies, train_ratings = shuffle(train_users, train_movies, train_ratings)

        for j in range(batches_per_epoch):
            users = train_users[j * batch_size: (j + 1) * batch_size]
            movies = train_movies[j * batch_size: (j + 1) * batch_size]
            targets = train_ratings[j * batch_size: (j + 1) * batch_size]

            users = torch.from_numpy(users).long()
            movies = torch.from_numpy(movies).long()
            targets = torch.from_numpy(targets)

            targets = targets.view(-1, 1).float()

            users, movies, targets = users.to(device), movies.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(users, movies)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        test_loss = []

        for j in range(int(np.ceil(len(test_users) / batch_size))):
            users = test_users[j * batch_size: (j + 1) * batch_size]
            movies = test_movies[j * batch_size: (j + 1) * batch_size]
            targets = test_ratings[j * batch_size: (j + 1) * batch_size]

            users = torch.from_numpy(users).long()
            movies = torch.from_numpy(movies).long()
            targets = torch.from_numpy(targets)

            targets = targets.view(-1, 1).float()

            users, movies, targets = users.to(device), movies.to(device), targets.to(device)

            outputs = model(users, movies)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())

        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0

        print(f'Epoch: {it + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}')

    return train_losses, test_losses


user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)

Ntrain = int(0.8 * len(ratings))

train_users = user_ids[: Ntrain]
train_movies = movie_ids[: Ntrain]
train_ratings = ratings[:Ntrain]

test_users = user_ids[Ntrain:]
test_movies = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

train_data = (train_users, train_movies, train_ratings)
test_data = (test_users, test_movies, test_ratings)

train_losses, test_losses = batch_gd2(model, criterion, optimizer, train_data, test_data, 10)

# plot losses
plt.figure()
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# make predictions
watched_movie_ids = df[df.new_user_id == 1].new_movie_id.values
potential_movie_ids = df[~df.new_movie_id.isin(watched_movie_ids)].new_movie_id.unique()
user_id_to_recommend = np.ones_like(potential_movie_ids)

t_user_ids = torch.from_numpy(user_id_to_recommend).long().to(device)
t_movie_ids = torch.from_numpy(potential_movie_ids).long().to(device)

with torch.no_grad():
    predictions = model(t_user_ids, t_movie_ids)

predictions_np = predictions.cpu().numpy().flatten()
sort_idx = np.argsort(-predictions_np)

top_ten_movie_ids = potential_movie_ids[sort_idx[: 10]]
top_ten_ratings = predictions_np[sort_idx[: 10]]

for movie, rating in zip(top_ten_movie_ids, top_ten_ratings):
    print('Movie:', movie, 'Rating:', rating)
