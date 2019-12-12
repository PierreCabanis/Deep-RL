import gym
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    cartpole_random()
