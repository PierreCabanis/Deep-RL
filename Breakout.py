import gym
import matplotlib.pyplot as plt
from DeepNet import ConvNet
import torch
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers import Monitor
import numpy as np
from DQN import DQN

torch.manual_seed(0)
FIRE = 1

config = {
    "TEST_MODE": True,     # Désactive l'entrainement et teste directement le réseau sauvegardé
    "DUELING_DQN": False,   # Active le dueling DQN
    "DOUBLE_DQN": True,     # Active le double DQN
    "BOLTZMANN": False,     # Exploration boltzmann (True), epsilon-greedy (False)
    "PLOTTING": False,      # Affichage du reward temps réel (lent)
    "RENDERING": False,     # Active l'affichage de l'env en temps réel (lent)
    "SAVE": False,          # Active la sauvegarde du DQN
    "SAVE_LOC": "eval_model.data",  # Nom du fichier de sauvegarde
    "N_TEST": 1             # Nombre de tests à réaliser (moyenne des récompenses)
}

param = {
    "BUFFER_SIZE": 2000,          # Taille de la mémoire
    "LR": 0.00025,                # Learning Rate
    "EPSILON": 1,                 # Paramètre pour epsilon greedy
    "EPSILON_MIN": 0.1,           # Epsilon min pour le decay
    "EPSILON_DECAY": 0.9999,      # epsilon = epsilon*epsilon_decay à chaque step
    "TAU": 1,                     # Parametre pour exploration Boltzmann
    "BATCH_SIZE": 32,             # Taille du Batch
    "GAMMA": 0.99,                # Paramètre prise en compte récompense
    "ALPHA": 0.005,               # Pour le double DQN
    "N_EPISODE": 1000,            # Nombre d'épisodes d'entrainements
    "N_STEP": 100,                # Nombre de step par épisode (Cartpool uniquement)
    "START_TRAIN": 1000,          # Démarre l'apprentissage après le nb de step
}


def test(env, dqn):
    """Lance un épisode en utilisant le dqn passé en paramètre et renvoie le score obtenu"""
    observation = env.reset()
    score = 0
    done = False
    env.step(FIRE)  # Fire at start
    lives = 5
    while not done:
        env.render()

        action = dqn.get_action(observation, test=True)
        observation, reward, done, info = env.step(action)
        score += reward

        if env.env.ale.lives() != lives:
            lives -= 1
            env.step(FIRE)  # Fire at start

    return score


def train(env, dqn):
    observation = env.reset()
    score = 0
    done = False
    env.step(FIRE)
    lives = 5

    # Tant que la partie n'est pas terminé
    while not done:

        if config["RENDERING"]: env.render()

        action = dqn.get_action(observation)
        observation_next, reward, done, info = env.step(action)

        if env.env.ale.lives() != lives:  # Perd une vie
            lives -= 1
            reward = -10
            env.step(FIRE)  # Relance la balle

        dqn.store(observation, action, observation_next, reward, done)
        observation = observation_next

        score += reward
        dqn.learn()

    return score, dqn


def breakout():
    """Lance l'entrainement ou/et le test d'un DQN selon la configuration sélectionnée"""

    # Création de l'environnement + Wrappers
    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, 4)
    env = Monitor(env, directory="Save")

    # Création du DQN
    if config["DUELING_DQN"]:
        dqn = DQN(Net=Net_Dueling,
                  param=param,
                  config=config,
                  n_action=env.action_space.n,
                  state_shape=[4, 84, 84])
    else:
        dqn = DQN(Net=ConvNet,
                  param=param,
                  config=config,
                  n_action=env.action_space.n,
                  state_shape=[4, 84, 84])
    # Boucle sur n episode
    scores_list = []
    if not config["TEST_MODE"]:
        for episode in range(param["N_EPISODE"]):

            score, dqn = train(env, dqn)

            scores_list.append(score)
            print("Episode : ", episode, " : Step : ", scores_list[-1])

            if episode % 20 == 0 and config["SAVE"]:   # Sauvegarde du DQN
                print("Saved !")
                torch.save(dqn.eval_model.state_dict(), "Save/"+config["SAVE_LOC"])

            if config["PLOTTING"]: plot_evolution(scores_list)

    # Sauvegarde du DQN
    if config["SAVE"]: torch.save(dqn.eval_model.state_dict(), "Save/"+config["SAVE_LOC"])

    # Phase de test
    score = []
    for k in range(config["N_TEST"]):
        s = test(env, dqn)
        score.append(s)
        print("Test Episode ", k+1, " : ", s)

    print("AVG : ", np.mean(score))
    print("STD : ", np.std(score))
    env.close()


def plot_evolution(data):
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(data)
    plt.grid()
    plt.pause(0.001)


if __name__ == '__main__':
    breakout()
