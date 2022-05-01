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
    n = 10
    z = torch.randn(n, 13)
    # z = torch.ones(n, 13)*-1
    # loop through z and change the nth element to 1
    # for i in range(n):
    #     z[i, i] = 1
    # z = torch.tensor([[-1,  -1, 1.5748, 0.4652,  0.1620, 0.7173,  2.1896,  0.5219,
    #     1.1290,  1.2080, 1.3345, 0.4526, 0.1958]]) # 1
    # z = torch.tensor([[0.7,  0.3, -1.5748, -1,  0.1620, 0.7173,  2.1896,  0.5219,
    #     1.1290,  1.2080, 1.3345, 0.4526, 0.1958]])
    # z = torch.tensor([[0.7,  0.3, 1.5748, 1,  -1.620, -1.73,  2.1896,  0.5219,
    #     1.1290,  1.2080, 1.3345, 0.4526, 0.1958]])
    # z = torch.ones(1, 13)
    print(z)
    sample = vae.decoder(z)
    save_image(sample.view(n, 1, 28, 28), 'gen.png')
    arr = z.cpu().numpy().tolist()
    for ar in arr:
        count = 0
        for i in maps:
            ar.insert(i, 0)
        while count != 100:
            #Generate random number between -1 and 1 of size 20
            # arr = np.random.uniform(-1, 1, 20)
            # arr[19] = np.random.random()
            obs, reward, done, info = env.step(ar)
            env.render()
            env.viewer.cam.elevation = 90
            env.viewer.cam.azimuth = 90
            env.viewer.cam.distance = 0.7
            count += 1

