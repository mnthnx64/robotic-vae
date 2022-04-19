import torch
from models import VAE
from torchvision.utils import save_image
import gym
from time import time, sleep
import numpy as np

# Load the trained encoder
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=13)
vae.load_state_dict(torch.load('runs/train/model.pth'))


env = gym.make('HandReach-v0')
obs = env.reset()
maps = [0, 1, 2, 5, 8, 11, 12]


# Draw the number
with torch.no_grad():
    # z = torch.randn(1, 13)
    z = torch.tensor([[0.5863,  0.3081, 1.5748, 0.4652,  0.1620, 0.7173,  2.1896,  0.5219,
        1.1290,  1.2080, 1.3345, 0.4526, 0.1958]])
    print(z)
    sample = vae.decoder(z)
    arr = z.cpu().numpy()[0].tolist()
    count = 0
    for i in maps:
        arr.insert(i, 0)
    while count != 100:
        #Generate random number between -1 and 1 of size 20
        # arr = np.random.uniform(-1, 1, 20)
        # arr[19] = np.random.random()
        obs, reward, done, info = env.step(arr)
        env.render()
        count += 1
    save_image(sample.view(1, 1, 28, 28), 'gen.png')

