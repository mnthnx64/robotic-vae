import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchviz

class VAE(nn.Module):
    """
    TODO: 
    - Construct the encoder to output 20x1 array
    - Same application for decoder.
    """
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        
        # Clip output of fc32 between limits -1 to 1
        self.fc32.weight.data.uniform_(-1, 1)

        # self.activation = 
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h1 = self.fc31(h)
        h2 = self.fc32(h)
        return h1, h2 # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return z

class Classifier(nn.Module):
    """Classify inputs into their corresponding numbers
    It takes an input tensor and outputs which number it belongs to
    """
    def __init__(self, inp_dim, out_dim):
        super(Classifier, self).__init__()

        self.l1 = nn.Linear(inp_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return F.log_softmax(x, dim=1)