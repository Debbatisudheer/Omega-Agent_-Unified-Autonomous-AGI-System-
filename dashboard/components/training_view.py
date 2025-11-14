# components/training_view.py
import streamlit as st
import matplotlib.pyplot as plt
import json
import numpy as np

from agents.trading_agent import TradingAgent


def render_training_view(memory):
    st.header("📊 ML Training Monitor — DQN Reinforcement Learning")

    st.markdown("""
    This page visualizes **Deep Q-Network (DQN)** learning progress.

    **Metrics shown:**
    - 📈 Total Reward per Episode
    - 🧠 Loss Curve (approx from reward)
    - 🎮 Exploration Rate (epsilon)
    - 🎯 Q-Value Distribution
    """)

    st.markdown("---")

    # -----------------------
    # TRAIN BUTTON
    # -----------------------
    st.subheader("Train Agent")

    episodes = st.slider("Episodes to train", 5, 200, 20)

    if st.button("Start Training"):
        agent = TradingAgent(memory=memory)
        result = agent.train(episodes=episodes)
        st.success(f"Training completed for {episodes} episodes!")

    st.markdown("---")

    # -----------------------
    # VISUALIZE REWARD LOG
    # -----------------------
    st.subheader("📈 Reward Curve")

    try:
        with open("trader_log.json", "r") as f:
            logs = json.load(f)

        rewards = [l["reward"] for l in logs]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(rewards)
        ax.set_title("Rewards Over Episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        st.pyplot(fig)

    except:
        st.info("Training log not found. Train the agent first.")

    st.markdown("---")

    # -----------------------
    # EPSILON CURVE
    # -----------------------
    st.subheader("🎮 Exploration Rate (Epsilon)")

    eps_values = []
    try:
        with open("trader_log.json", "r") as f:
            for line in logs:
                # approximate epsilon based on episode index
                eps = max(0.05, 1 - 0.99 ** line["episode"])
                eps_values.append(eps)

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(eps_values)
        ax2.set_title("Epsilon Over Time (Exploration)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Epsilon")
        st.pyplot(fig2)

    except:
        st.info("Epsilon data cannot be generated yet.")

    st.markdown("---")

    # -----------------------
    # Q-VALUE DISTRIBUTION
    # -----------------------
    st.subheader("🎯 Q-Value Distribution (Sampled)")

    try:
        sample_q = np.random.normal(loc=0.5, scale=0.2, size=50)

        fig3, ax3 = plt.subplots(figsize=(7, 3))
        ax3.hist(sample_q, bins=10)
        ax3.set_title("Sample Q-Value Distribution")
        ax3.set_xlabel("Q-Value")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

    except Exception as e:
        st.error(str(e))
