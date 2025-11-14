# components/trading_view.py
import streamlit as st
import matplotlib.pyplot as plt
import json

from agents.trading_agent import TradingAgent

def render_trading_view(memory):
    st.header("Trading Monitor")

    if st.button("Run One Trading Step"):
        agent = TradingAgent(memory=memory)
        out = agent.step()
        st.write(out)

    st.markdown("---")
    st.subheader("Reward Log")

    try:
        with open("trader_log.json", "r") as f:
            logs = json.load(f)

        rewards = [l["reward"] for l in logs]

        plt.figure(figsize=(6,3))
        plt.plot(rewards)
        plt.title("Rewards Over Episodes")
        st.pyplot(plt)
    except:
        st.info("No reward log found. Train the agent first.")
