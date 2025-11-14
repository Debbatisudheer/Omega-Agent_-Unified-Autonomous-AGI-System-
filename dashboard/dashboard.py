# dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from components.sidebar import render_sidebar
from components.memory_view import render_memory_view
from components.research_view import render_research_view
from components.trading_view import render_trading_view
from components.planner_view import render_planner_view
from components.training_view import render_training_view

from memory.vector_memory import VectorMemory
from agents.planner_agent import run_planner
from agents.trading_agent import run_trading_agent
from agents.research_agent import run_research_agent


# ===========================
# MEMORY INITIALIZATION
# ===========================
@st.cache_resource
def get_memory():
    try:
        mem = VectorMemory()
        return mem
    except Exception as e:
        st.error(f"Failed to initialize VectorMemory: {e}")
        return None

memory = get_memory()


# ===========================
# PAGE CONFIG + SIDEBAR
# ===========================
st.set_page_config(page_title="Omega Dashboard", layout="wide")
render_sidebar()


# ===========================
# TABS
# ===========================
tabs = st.tabs([
    "Home",
    "Memory Explorer",
    "Research Lab",
    "Trading Monitor",
    "ML Training Monitor",
    "Planner",
    "System Info"
])


# ===========================
# HOME TAB
# ===========================
with tabs[0]:
    st.header("🌌 Omega Autonomous Agent — Home")

    # --------- ENVIRONMENT WARNINGS ---------
    openai_missing = not os.getenv("OPENAI_API_KEY")
    pinecone_missing = not os.getenv("PINECONE_API_KEY")

    if openai_missing or pinecone_missing:
        st.markdown("### ⚠️ System Notice")
        st.info("""
        🚫 **API keys are disabled by admin (Sudheer).**

        Some features may not work:
        - ❌ Research summaries (OpenAI API disabled)
        - ❌ Pinecone vector memory (disabled)

        But these features still work:
        - ✅ Planner Agent (Daily schedule)
        - ✅ Trading Agent (DQN RL)
        - ✅ Local Memory Mode
        - ✅ Full Dashboard UI
        """)

    # --------- PROJECT OVERVIEW ---------
    st.markdown("""
    ## 🧠 What is Omega Agent?

    Omega is a **multi-agent autonomous AI system** that can:

    - 📅 Plan your day automatically  
    - 📈 Trade using Reinforcement Learning (DQN)  
    - 🔬 Fetch and summarize latest AI papers  
    - 🧠 Store long-term knowledge (Pinecone memory)  
    - ⚙️ Run manually OR fully autonomous  

    Omega = Planner + Trading + Research + Memory → **One Unified Dashboard**
    """)

    st.markdown("---")

    # --------- MANUAL CONTROLS ---------
    st.markdown("## ⚡ Manual Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Planner"):
            run_planner(memory=memory)
            st.success("Planner ran successfully.")

    with col2:
        if st.button("Trading Step"):
            result = run_trading_agent(memory=memory)
            st.write(result)

    with col3:
        if st.button("Research Cycle"):
            result = run_research_agent(memory=memory)
            st.write(result)



# ===========================
# MEMORY TAB
# ===========================
with tabs[1]:
    render_memory_view(memory)


# ===========================
# RESEARCH TAB
# ===========================
with tabs[2]:
    render_research_view(memory)


# ===========================
# TRADING TAB
# ===========================
with tabs[3]:
    render_trading_view(memory)


# ===========================
# ML TRAINING MONITOR TAB
# ===========================
with tabs[4]:
    render_training_view(memory)


# ===========================
# PLANNER TAB
# ===========================
with tabs[5]:
    render_planner_view(memory)


# ===========================
# SYSTEM INFO TAB
# ===========================
with tabs[6]:
    st.subheader("System Info")
    st.write({
        "Memory backend": type(memory).__name__ if memory else "None",
        "Root Directory": os.getcwd(),
        "Dashboard Version": "1.1 (includes ML Monitor)"
    })
