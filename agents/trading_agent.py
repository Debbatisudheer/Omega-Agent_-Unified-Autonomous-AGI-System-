# agents/trading_agent.py

import os
import json
from core.rl_utils import SimpleMarketEnv, DQNAgent
from memory.vector_memory import VectorMemory
import numpy as np

MODEL_FILE = "trader_model.pth"
TRAIN_LOG = "trader_log.json"

class TradingAgent:
    def __init__(self, memory: VectorMemory = None):
        self.memory = memory
        self.env = SimpleMarketEnv(seq_len=300, volatility=0.01)
        self.state_dim = 9  # returns length (10->diff yields 9)
        self.agent = DQNAgent(state_dim=self.state_dim, action_dim=3)
        self.episode = 0
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(MODEL_FILE):
            try:
                data = np.load(MODEL_FILE, allow_pickle=True).item()
                self.agent.net.load_state_dict(data["net"])
                print("Loaded trader model.")
            except Exception as e:
                print("Failed to load model:", e)

    def train(self, episodes=50):
        log = []
        for ep in range(episodes):
            s = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                a = self.agent.act(s)
                ns, r, done, _ = self.env.step(a)
                self.agent.store(s, a, r, ns, done)
                loss = self.agent.learn()
                s = ns
                total_reward += r
                steps += 1

            self.episode += 1
            log.append({"episode": self.episode, "reward": float(total_reward), "steps": steps})

            if self.memory:
                self.memory.add(
                    f"Trader ep {self.episode} reward {total_reward}",
                    metadata={"type": "trader_training"}
                )

            print(f"Train ep {self.episode} reward {total_reward:.4f} eps {self.agent.eps:.3f}")

        with open(TRAIN_LOG, "w") as f:
            json.dump(log, f, indent=2)
        return log

    def step(self):
        """
        Executes a single trading decision (buy/hold/sell)
        """
        s = self.env.reset()
        a = self.agent.act(s)

        action_map = {0: "hold", 1: "buy", 2: "sell"}
        decision = {"action": action_map[a], "eps": float(self.agent.eps)}

        # Save to memory
        if self.memory:
            self.memory.add(
                "Trading decision: " + decision["action"],
                metadata={"type": "trading_decision"}
            )

        # Train a bit initially
        if self.episode < 5:
            self.train(episodes=5)

        return decision


# ============================================================
# WRAPPER FUNCTION — REQUIRED BY omega.py
# ============================================================

def run_trading_agent(memory=None):
    """
    Wrapper for Omega system.
    Accepts optional VectorMemory and returns clean trading action.
    """
    agent = TradingAgent(memory=memory)
    result = agent.step()
    return {
        "action": result["action"]
    }

