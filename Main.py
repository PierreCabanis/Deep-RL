import gym
import matplotlib.pyplot as plt
from DeepNet import Net
import torch
from time import sleep
from Buffer import Buffer
from random import random

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

def cartpole_NN(epsilon):
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = Net(observation_space, action_space)
    memory = Buffer(100000)

    learn_start = 10
    batch_size = 10
    tau = 0.5


    for episode in range(100):
        observation = env.reset()
        for t in range(250):
            env.render()
            Q = model.forward(torch.Tensor(observation))

            if episode>learn_start:
                batch = memory.get_batch(batch_size)
                rewards = torch.tensor([b[3] for b in batch])
                obs = torch.Tensor([b[0] for b in batch])
                acts = [b[1] for b in batch]

                Q_train = model.forward(obs)[:,acts]
                print(Q_train)

            Q = model.forward(torch.Tensor(observation))
            action = torch.argmax(Q).item()

            observation_next, reward, done, info = env.step(action)
            memory.append([observation, action, observation_next, reward, done])
            observation = observation_next

            if done:
                if episode>learn_start:
                    loss = Q_train - rewards
                print("Episode ", episode+1, ", C'est mort.")
                break
    env.close()


if __name__ == '__main__':
    #cartpole_random()
    cartpole_NN(0.1)