import gym
import matplotlib.pyplot as plt
import torch
import numpy as np

from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

from DQN import DQN
from DeepNet import ConvNet

param = {
    "BUFFER_SIZE": 2000,
    "LR": 1e-3,
    "EPSILON": 0.9,
    "EPSILON_MIN": 0.1,
    "EPSILON_DECAY": 0.995,
    "BATCH_SIZE": 32,
    "GAMMA": 0.9,
    "ALPHA": 0.005,
    "N_EPISODE": 1000,
    "N_STEP": 100,
    "START_TRAIN": 1000,
}


def breakout():
    # Create environment
    env = gym.make('BreakoutNoFrameskip-v4')
    # Wrappers
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, 4)

    action_space = env.action_space.n
    torch.manual_seed(0)

    dqn = DQN(ConvNet, param, action_space, [4, 84, 84])

    steps = []
    for episode in range(param["N_EPISODE"]):
        observation = env.reset()
        steps.append(0)
        done = False
        while not done:
            # env.render()
            action = dqn.get_action(observation)
            observation_next, reward, done, info = env.step(action)
            dqn.store(observation, action, observation_next, reward, done)
            observation = observation_next
            steps[-1] += reward
            dqn.learn()
        print("Episode : ", episode, " : Step : ", steps[-1])
        if episode % 50 == 0:
            print("Saved !")
            torch.save(dqn.eval_model.state_dict(), "Save/eval_model.data")

    torch.save(dqn.eval_model.state_dict(), "Save/eval_model.data")

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
    plt.plot(ret[n - 1:] / n)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    breakout()
