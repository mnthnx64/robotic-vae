import torch
from models import VAE, Classifier
from torchvision.utils import save_image
import gym
from time import time, sleep
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=13)
vae.load_state_dict(torch.load('runs/train/model.pth'))
cl = Classifier(13, 5)
cl.load_state_dict(torch.load('runs/train/modelClassifier.pth'))

env = gym.make('HandReach-v0')
obs = env.reset()
maps = [0, 1, 2, 5, 8, 11, 12]
# root window
root = tk.Tk()
root.geometry('300x200')
root.resizable(False, False)
root.title('HandReach')

values = [tk.DoubleVar(value=1) for i in range(5)]
map_joints = {
    0: [0,1],
    1: [2,3],
    2: [4,5],
    3: [6,7],
    4: [8,9],
}

def actuate_hand():
    z = torch.zeros((1, 13))
    for key in map_joints.keys():
        for i in map_joints[key]:
            z[0, i] = values[key].get()
    with torch.no_grad():
        sample = vae.decoder(z)
        out = cl(z)
        print(torch.argmax(out, dim=1))
        # save_image(sample.view(1, 1, 28, 28), 'gen.png')
        plt.imshow(sample.view(1, 1, 28, 28).numpy()[0, 0, :, :])
        # plt.show()
        arr = z.numpy().tolist()
        for ar in arr:
            count = 0
            for i in maps:
                ar.insert(i, 0)
            #Generate random number between -1 and 1 of size 20
            # arr = np.random.uniform(-1, 1, 20)
            # arr[19] = np.random.random()
            obs, reward, done, info = env.step(ar)
            env.render()
            # env.viewer.cam.elevation = 90
            # env.viewer.cam.azimuth = 90
            # env.viewer.cam.distance = 0.7

    root.after(100, actuate_hand)


ttk.Label(root, text='Adjust the values of the hand').grid(row=0, column=0)
for idx, i in enumerate(values):
    ttk.Label(root,text=f'Finger{idx+1}:').grid(column=0,row=idx+1,sticky='w')
    x = ttk.Scale(root, from_=-1, to=1, orient=tk.HORIZONTAL, variable=i)
    x.grid(row=idx+1, column=1, sticky='we')

root.after(100, actuate_hand)
root.mainloop()