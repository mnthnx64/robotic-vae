import os
from pathlib import Path
import shutil
from tkinter import TRUE

import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from models import VAE, Classifier
from torchvision.utils import save_image

BATCH_SIZE = 100
DATASET_DIR = './dataset/mnist_data/'
SAVE_DIR = 'runs/train/'
RESUME = True

if os.path.exists(SAVE_DIR) and not RESUME: shutil.rmtree(SAVE_DIR)
if not RESUME: os.makedirs(SAVE_DIR)

# MNIST Dataset
# train_dataset = datasets.MNIST(root=DATASET_DIR, train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, transform=transforms.ToTensor(), download=False)
# dataset = datasets.MNIST(root='./data')


train_dataset = datasets.MNIST(root=DATASET_DIR, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root=DATASET_DIR, train=False, transform=transforms.ToTensor(), download=False)
train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if train_dataset[i][1] < 5]
test_dataset = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i][1] < 5]
    # return train_dataset, test_dataset

# idx = train_dataset.train_labels <= 5

# train_dataset.train_labels = train_dataset.train_labels[idx]
# train_dataset.train_data = train_dataset.train_data[idx]

# idx = test_dataset.test_labels <= 5
# test_dataset.test_labels = test_dataset.test_labels[idx]
# test_dataset.test_data = test_dataset.test_data[idx]
# test_dataset.train_labels = test_dataset.train_labels[idx]
# test_dataset.train_data = test_dataset.train_data[idx]

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)



vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=13)
cl = Classifier(13, 5)

if RESUME:
    vae.load_state_dict(torch.load(SAVE_DIR + "model.pth"))


optimizer = optim.Adam(vae.parameters())
clOptimizer = optim.Adam(cl.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# Get Classifier loss function
def classifierLoss(pred, target):
    return F.cross_entropy(pred, target)

def trainVAE(epoch):
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
        torch.save(vae.state_dict(), path) # saving model
        with torch.no_grad():
            z = torch.randn(64, 13)
            sample = vae.decoder(z)
            
            save_image(sample.view(64, 1, 28, 28), SAVE_DIR + 'Sample_%s.png' % epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

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



def trainClassifier():
    cl.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        clOptimizer.zero_grad()
        
        z = vae.encode(data)
        preds = cl(z)
        loss = classifierLoss(preds, _)
        
        loss.backward()
        train_loss += loss.item()
        clOptimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    if (epoch % 10) == 0:
        path = os.path.join(SAVE_DIR, 'modelClassifier.pth')
        torch.save(cl.state_dict(), path) # saving model
        with torch.no_grad():
            z = torch.randn(1, 13)
            sample = cl(z)
            print(z, sample)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    cl.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data
            z = vae.encode(data)
            pred = cl(z)
            
            # sum up batch loss
            test_loss += classifierLoss(pred, _).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# for epoch in range(1, 100):
#     trainVAE(epoch)

for epoch in range(1, 100):
    trainClassifier()
