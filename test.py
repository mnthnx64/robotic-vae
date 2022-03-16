from time import time
import numpy as np
import gym


env = gym.make('HandReach-v0')
obs = env.reset()
done = False

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()


arr = np.zeros((20))
count = 0

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

while count != 1000:
    action = policy(obs['observation'], obs['desired_goal'])
    # arr[0] = np.random.randn() 
    # arr[1] = np.random.randn()
    # arr[2] = np.random.randn()
    arr[12] = np.random.randn()
    # arr[4] = np.random.randn()
    obs, reward, done, info = env.step(arr)
    env.render()
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    # print('reward is {}, substitute_reward is {}'.format(
    #     reward, substitute_reward))
    count += 1

env.close()