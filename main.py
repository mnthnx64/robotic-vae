import os
from pathlib import Path
import shutil

import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from models import VAE
from torchvision.utils import save_image

BATCH_SIZE = 100
DATASET_DIR = './dataset/mnist_data/'
SAVE_DIR = 'runs/train/'
RESUME = True

if os.path.exists(SAVE_DIR) and not RESUME: shutil.rmtree(SAVE_DIR)
if not RESUME: os.makedirs(SAVE_DIR)

# MNIST Dataset
train_dataset = datasets.MNIST(root=DATASET_DIR, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
if RESUME:
    vae.load_state_dict(torch.load(SAVE_DIR + "model.pth"))
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    if (epoch % 10) == 0:
        path = os.path.join(SAVE_DIR, 'model.pth')
        torch.save(vae.cpu().state_dict(), path) # saving model
        with torch.no_grad():
            z = torch.randn(64, 2)
            sample = vae.decoder(z)
            
            save_image(sample.view(64, 1, 28, 28), SAVE_DIR + 'Sample_%s.png' % epoch)
        if torch.cuda.is_available():
            vae.cuda()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, 201):
    train(epoch)
    test()
