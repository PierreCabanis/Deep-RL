import gym
import matplotlib.pyplot as plt
from DeepNet import ConvNet
import torch
import random
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from numpy.random import seed
import numpy as np
from time import time
from DQN import DQN


param = {
    "BUFFER_SIZE": 10000,
    "LR": 1e-2,
    "EPSILON": 0.9,
    "N_STEP": 1000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.9,
    "ALPHA" : 0.005,
    "N_EPISODE": 5000,
    "START_TRAIN": 1000
}

def breakout():
    env = gym.make('BreakoutNoFrameskip-v4')
    # Wrappers
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, 4)

    action_space = env.action_space.n
    seed(0)
    random.seed(0)
    torch.manual_seed(0)

    dqn = DQN(ConvNet, param, action_space, [4,84,84])
    steps = []
    for episode in range(param["N_EPISODE"]):
        observation = env.reset()
        steps.append(0)
        done = False
        t = time()
        while not done:
            #env.render()
            action = dqn.get_action(observation)
            observation_next, reward, done, info = env.step(action)
            dqn.store(observation, action, observation_next, reward, done)
            observation = observation_next
            steps[-1] += reward
            if dqn.memory.index > param["BATCH_SIZE"]:
                dqn.learn()

        plot_evolution(steps)

    observation = env.reset()
    steps.append(0)
    done = False
    while not done:
        env.render()
        action = dqn.get_action(observation, top=True)
        observation, reward, done, info = env.step(action)
        steps[-1] += reward
    print(steps[-1])
    env.close()

def plot_evolution(data):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(data)
    plt.grid()
    ret = np.cumsum(data)

    n = 20
    ret[n:] = ret[n:] - ret[:-n]
    plt.plot(ret[n-1:]/n)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    breakout()