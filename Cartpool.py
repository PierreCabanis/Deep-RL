import gym
import matplotlib.pyplot as plt
from DeepNet import Net
import torch
import random
from numpy.random import seed
import numpy as np
from DQN import DQN
from time import time

param = {
    "BUFFER_SIZE": 10000,
    "LR": 1e-2,
    "TAU": 1,
    "UPDATE_MODEL_STEP": 1000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.9,
    "N_EPISODE": 5000,
    "START_TRAIN": 1000
}


def cartpole_random():
    env = gym.make('CartPole-v0')

    env.reset()
    rewards = []
    for t in range(250):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

    plt.ylabel("Rewards")
    plt.xlabel("Nb interactions")
    plt.plot(rewards)
    env.close()
    plt.grid()
    plt.show()


def cartpole_NN():
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    seed(0)
    random.seed(0)
    torch.manual_seed(0)

    dqn = DQN(Net, param, action_space, [observation_space], cuda=False)
    steps = []

    for episode in range(10000):
        observation = env.reset()
        done = False
        steps.append(0)
        t = time()
        for i in range(100):
            #env.render()
            action = dqn.get_action(observation)
            observation_next, reward, done, info = env.step(action)

            dqn.store(observation, action, observation_next, reward, done)

            observation = observation_next
            dqn.learn()
            steps[-1] += reward
        print('T : ', time() - t)
        if episode %10:
            plot_evolution(steps)
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
    # cartpole_random()
    cartpole_NN()
