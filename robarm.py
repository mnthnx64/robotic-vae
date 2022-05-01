import gym
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import glfw
import imageio

env = gym.make('HandReach-v0')
obs = env.reset()
done = False

def policy(observation, desired_goal):
    return env.action_space.sample()


"""
Array:
0: X-arm
1: Y-arm
2: X-index
3: Y-index
4: Z-index
5: X-middle
6: Y-middle
7: Z-middle
8: X-ring
9: Y-ring
10: Z-ring
11: Palm-pinky displacement
12: X-pinky
13: Y-pinky + Y-ring
14: Z-pinky + Z-ring
15: Rot-thumb
16: Y-thumb
17: X-thumb
18: Z-thumb.j1
19: Z-thumb.j2
"""
maps = [0, 1, 2, 5, 8, 11, 12, 17, 18, 19]
arr = np.ones((20)) * -1
count = 0

arr = np.ones((20))*-1
# arr[3] = 0
# arr[4] = 0

while count != 1000:
    action = policy(obs['observation'], obs['desired_goal'])
    #Generate random number between -1 and 1 of size 20
    arr = np.ones(20)
    # arr[15] = 1
    # arr[19] = 1
    arr[3]= -1
    arr[4]=-1
    obs, reward, done, info = env.step(arr)
    
    # Render the environment in mode rgb_array
    # Save the rendered environment to a file
    # env.env.render(mode='rgb_array')


    env.render()
    # env.viewer.cam.elevation = 90
    # env.viewer.cam.azimuth = 90
    # env.viewer.cam.distance = 0.7

    
    count += 1


env.close()