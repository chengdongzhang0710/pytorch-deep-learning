import torch
import torch.nn as nn
import numpy as np

# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

# generate data
N = 2
T = 10
D = 3
M = 5
K = 2
X = np.random.randn(N, T, D)


# create model
class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            nonlinearity='tanh',
            batch_first=True,
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(1, X.size(0), self.M)  # one rnn layer
        out, _ = self.rnn(X, h0)
        out = self.fc(out)
        return out


model = SimpleRNN(n_inputs=D, n_hidden=M, n_outputs=K)

# calculate output
inputs = torch.from_numpy(X.astype(np.float32))
output = model(inputs)

Yhats_torch = output.detach().numpy()

W_xh, W_hh, b_xh, b_hh = model.rnn.parameters()
W_xh = W_xh.data.numpy()
b_xh = b_xh.data.numpy()
W_hh = W_hh.data.numpy()
b_hh = b_hh.data.numpy()

Wo, bo = model.fc.parameters()
Wo = Wo.data.numpy()
bo = bo.data.numpy()

# replicate output
# h_last = np.zeros((N, M))
# Yhats = np.zeros((N, T, K))
# for t in range(T):
#     for n in range(N):
#         h = np.tanh(X[n][t].dot(W_xh.T) + b_xh + h_last[n].dot(W_hh.T) + b_hh)
#         y = h.dot(Wo.T) + bo
#         Yhats[n][t] = y
#         h_last[n] = h

h_last = np.zeros((N, M))
Yhats = np.zeros((N, T, K))
for t in range(T):
    h = np.tanh(X[:, t].dot(W_xh.T) + np.repeat(b_xh[np.newaxis, :], N, axis=0) +
                h_last.dot(W_hh.T) + np.repeat(b_hh[np.newaxis, :], N, axis=0))
    y = h.dot(Wo.T) + np.repeat(bo[np.newaxis, :], N, axis=0)
    Yhats[:, t] = y
    h_last = h

print(np.allclose(Yhats, Yhats_torch))
