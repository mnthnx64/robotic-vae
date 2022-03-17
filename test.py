import torch
from models import VAE
from torchvision.utils import save_image

# Load the trained encoder
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
vae.load_state_dict(torch.load('runs/train/model.pth'))

with torch.no_grad():
    z = torch.randn(10, 2)
    print(z)
    sample = vae.decoder(z)
    save_image(sample.view(10, 1, 28, 28), 'gen.png')