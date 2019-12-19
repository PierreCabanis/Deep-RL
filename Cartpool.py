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
    "LR": 25e-4,
    "EPSILON": 1,
    "EPSILON_MIN" : 0.1,
    "EPSILON_DECAY" : 0.9/1000000,
    "N_STEP": 1000,
    "BATCH_SIZE": 32,
    "GAMMA": 0.99,
    "ALPHA" : 0.005,
    "N_EPISODE": 100,
    "START_TRAIN": 500
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
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    seed(0)
    random.seed(0)
    torch.manual_seed(0)

    dqn = DQN(Net, param, action_space, [observation_space], cuda=False)
    steps = []

    for episode in range(param["N_EPISODE"]):
        observation = env.reset()
        steps.append(0)
        for k in range(param["N_STEP"]):
            env.render()
            action = dqn.get_action(observation)
            observation_next, reward, done, info = env.step(action)
            if done:
                reward = -1
            dqn.store(observation, action, observation_next, reward, done)

            steps[-1] += reward
            if dqn.memory.index > param["BATCH_SIZE"]:
                dqn.learn()

            observation = observation_next
        print("Episode : ", episode, " | Steps : ", steps[-1])
        plot_evolution(steps)

    observation = env.reset()
    steps.append(0)
    done = False
    while not done:
        env.render()
        action = dqn.get_action(observation, test=True)
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
    # cartpole_random()
    cartpole_NN()
