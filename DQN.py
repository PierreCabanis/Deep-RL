import torch
import torch.nn as nn
from numpy.random import choice
import torch.nn.functional as F
from random import sample


class DQN:
    def __init__(self, Net, param, n_action, n_state):
        use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if use_cuda else "cpu")
        # Création des NN d'éval et de target
        self.eval_model = Net(n_action, n_state).to(self.device)
        self.target_model = Net(n_action, n_state).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())

        self.param = param

        self.memory = Buffer(self.param["BUFFER_SIZE"])
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=self.param["LR"])
        self.criterion = nn.MSELoss().to(self.device)

        self.step_counter = 0


    def get_action(self,state):
        state = torch.FloatTensor(state).to(self.device)

        Q = self.eval_model(state)
        proba = F.softmax(Q / self.param["TAU"], dim=0).detach()
        action = choice([0, 1], p=proba.cpu().numpy().round(2))

        return action

    def store(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, reward, done])


    def learn(self):
        self.step_counter += 1
        if self.step_counter < self.param["START_TRAIN"]:
            return

        if self.step_counter % self.param["UPDATE_MODEL_STEP"] == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())


        batch = self.memory.get_batch(self.param["BATCH_SIZE"])
        state, action, next_state, reward, done = [], [], [], [], []
        for b in batch:
            state.append(b[0])
            action.append(b[1])
            next_state.append(b[2])
            reward.append(b[3])
            done.append(b[4])

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)

        Q_eval = self.eval_model(state).gather(1, action).reshape([self.param["BATCH_SIZE"]])
        Q_next = self.target_model(next_state).detach()
        Q_target = reward + self.param["GAMMA"]*Q_next.max(1)[0].reshape([self.param["BATCH_SIZE"]])*reward

        loss = self.criterion(Q_eval, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Buffer:
    def __init__(self, taille_buffer):
        self.content = []
        self.index = 0
        self.taille = taille_buffer

    def append(self, o):
        if self.index < len(self.content):
            self.content[self.index] = o
        else:
            self.content.append(o)

        self.index = (self.index + 1) % self.taille

    def get_batch(self, batch_size):
        return sample(self.content, batch_size)
