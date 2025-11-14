# core/rl_utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Simple price environment: random walk with trend; discretized observation.
class SimpleMarketEnv:
    def __init__(self, seq_len=200, volatility=0.01):
        # create a simulated price series
        self.seq_len = seq_len
        self.volatility = volatility
        self.reset()

    def reset(self):
        self.t = 0
        self.price = 100.0
        self.history = [self.price]
        return self._obs()

    def step(self, action):
        # action: 0 = hold, 1 = buy, 2 = sell
        # price evolves
        change = np.random.randn() * self.volatility * self.price
        self.price += change
        self.history.append(self.price)
        reward = 0.0
        # simple reward: if buy then reward positive when next step price goes up (simulated)
        if action == 1:
            reward = max(0, self.history[-1] - self.history[-2])
        elif action == 2:
            reward = max(0, self.history[-2] - self.history[-1])
        self.t += 1
        done = (self.t >= self.seq_len - 1)
        return self._obs(), reward, done, {}

    def _obs(self):
        # return last 10 returns as state
        hist = np.array(self.history[-10:])
        if len(hist) < 10:
            pad = np.ones(10 - len(hist)) * hist[0]
            hist = np.concatenate([pad, hist])
        returns = np.diff(hist) / hist[:-1]
        return returns.astype(np.float32)

# Simple DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim=3, hidden=64, lr=1e-3, gamma=0.99):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.target = nn.Sequential(*[l for l in self.net])  # copy
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = deque(maxlen=10000)
        self.batch_size = 64
        self.update_steps = 0
        self.eps = 1.0

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(0,3)
        s = torch.tensor(state).float().unsqueeze(0)
        q = self.net(s).detach().numpy()[0]
        return int(np.argmax(q))

    def store(self, s,a,r,ns,done):
        self.replay.append((s,a,r,ns,done))

    def learn(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)
        s,a,r,ns,done = zip(*batch)
        s = torch.tensor(np.array(s)).float()
        a = torch.tensor(a).long().unsqueeze(1)
        r = torch.tensor(r).float().unsqueeze(1)
        ns = torch.tensor(np.array(ns)).float()
        done = torch.tensor(done).float().unsqueeze(1)

        q = self.net(s).gather(1,a)
        with torch.no_grad():
            q_next = self.target(ns).max(1,keepdim=True)[0]
            q_target = r + (1-done) * self.gamma * q_next

        loss = nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 200 == 0:
            self._sync_target()
        # anneal epsilon
        self.eps = max(0.05, self.eps * 0.995)
        return loss.item()

    def _sync_target(self):
        self.target.load_state_dict(self.net.state_dict())
