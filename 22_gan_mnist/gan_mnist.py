import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from cv2 import imread
import os

# load data
# make image pixel values between -1 and +1 to obtain better effects
# ToTensor() returns values between 0 and 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

train_dataset = torchvision.datasets.MNIST(root='.', train=True, transform=transform, download=True)

batch_size = 128
data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# generative adversarial networks
# create model
# discriminator
D = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
)

# generator
latent_dim = 100
G = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(256, momentum=0.7),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(512, momentum=0.7),
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024, momentum=0.7),
    nn.Linear(1024, 784),
    nn.Tanh(),
)

# enable gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
D.to(device)
G.to(device)

criterion = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


# scale image back to (0, 1)
def scale_image(img):
    out = (img + 1) / 2
    return out


# create a folder to store generated images
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# train model
ones_ = torch.ones(batch_size, 1).to(device)
zeros_ = torch.zeros(batch_size, 1).to(device)
d_losses = []
g_losses = []
for epoch in range(200):
    for inputs, _ in data_loader:
        n = inputs.size(0)
        inputs = inputs.reshape(n, 784).to(device)
        ones = ones_[: n]
        zeros = zeros_[: n]

        # train discriminator
        # real images
        real_outputs = D(inputs)
        d_loss_real = criterion(real_outputs, ones)

        # fake images
        noise = torch.randn(n, latent_dim).to(device)
        fake_images = G(noise)
        fake_outputs = D(fake_images)
        d_loss_fake = criterion(fake_outputs, zeros)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        # train generator twice since generator is more sophisticated
        for _ in range(2):
            # fake images
            noise = torch.randn(n, latent_dim).to(device)
            fake_images = G(noise)
            fake_outputs = D(fake_images)

            g_loss = criterion(fake_outputs, ones)  # reverse labels
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch: {epoch + 1}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        fake_images = fake_images.reshape(-1, 1, 28, 28)
        save_image(scale_image(fake_images), f'gan_images/{epoch + 1}.png')

# plot loss
plt.figure()
plt.plot(d_losses, label='d_losses')
plt.plot(g_losses, label='g_losses')
plt.legend()
plt.show()

a = imread('gan_images/50.png')
plt.imshow(a)
plt.show()

a = imread('gan_images/100.png')
plt.imshow(a)
plt.show()

a = imread('gan_images/150.png')
plt.imshow(a)
plt.show()

a = imread('gan_images/200.png')
plt.imshow(a)
plt.show()
