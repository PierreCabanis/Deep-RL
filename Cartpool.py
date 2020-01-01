import gym
import matplotlib.pyplot as plt
from DeepNet import Net
import torch
import random
from numpy.random import seed
import numpy as np
from DQN import DQN


config = {
    "TEST_MODE": False,     # Désactive l'entrainement et teste directement le réseau sauvegardé
    "DUELING_DQN": False,   # Active le dueling DQN
    "DOUBLE_DQN": True,     # Active le double DQN
    "BOLTZMANN": False,     # Exploration boltzmann (True), epsilon-greedy (False)
    "PLOTTING": True,      # Affichage du reward temps réel (lent)
    "RENDERING": False,     # Active l'affichage de l'env en temps réel (lent)
    "SAVE": False,          # Active la sauvegarde du DQN
    "SAVE_LOC": "eval_model_cartpool0.data",  # Nom du fichier de sauvegarde
    "N_TEST": 10             # Nombre de tests à réaliser (moyenne des récompenses)
}

param = {
    "BUFFER_SIZE": 100000,
    "LR": 1e-4,
    "EPSILON": 1,
    "EPSILON_MIN": 0.1,
    "EPSILON_DECAY": 0.999,
    "BATCH_SIZE": 32,
    "GAMMA": 0.9,
    "ALPHA": 0.005,
    "N_EPISODE": 200,
    "N_STEP": 200,
    "START_TRAIN": 1000,
}

def test(env, dqn):
    observation = env.reset()
    score = 0
    done = False
    while not done:
        env.render()
        action = dqn.get_action(observation, test=True)
        observation, reward, done, info = env.step(action)
        score += reward
    return score


def train(env, dqn):
    observation = env.reset()
    score = 0
    done = False
    for k in range(param["N_STEP"]):
    #while not done:
        # env.render()
        action = dqn.get_action(observation)
        observation_next, reward, done, info = env.step(action)
        dqn.store(observation, action, observation_next, reward, done)

        score += reward
        if dqn.memory.index > param["BATCH_SIZE"]:
            dqn.learn()

        observation = observation_next
    return score, dqn


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

    dqn = DQN(Net, param, config, action_space, [observation_space])
    scores_list = []

    if not config["TEST_MODE"]:
        for episode in range(param["N_EPISODE"]):

            score, dqn = train(env, dqn)
            scores_list.append(score)

            print("Episode : ", episode, " | Steps : ", scores_list[-1])

            if episode % 20 == 0 and config["SAVE"]:  # Sauvegarde du DQN
                print("Saved !")
                torch.save(dqn.eval_model.state_dict(), "Save/" + config["SAVE_LOC"])

            if config["PLOTTING"]: plot_evolution(scores_list)

        # Sauvegarde du DQN
        if config["SAVE"] : torch.save(dqn.eval_model.state_dict(), "Save/" + config["SAVE_LOC"])


    score = []
    for k in range(config["N_TEST"]):
        s = test(env, dqn)
        score.append(s)
        print("Test Episode ", k + 1, " : ", s)

    print("AVG : ", np.mean(score))
    print("STD : ", np.std(score))
    env.close()


def plot_evolution(data):
    plt.figure(2)
    plt.clf()
    plt.title("Reward")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(data)
    plt.grid()
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    # cartpole_random()
    cartpole_NN()
    plt.show()
