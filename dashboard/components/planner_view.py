# components/planner_view.py
import streamlit as st
from agents.planner_agent import PlannerAgent

def render_planner_view(memory):
    st.header("Planner View")

    if st.button("Generate Today's Schedule"):
        agent = PlannerAgent(memory=memory)
        schedule = agent.plan_day()

        for s in schedule:
            st.write(f"{s['start']} → {s['end']}   |   {s['title']} ({s['est_minutes']} mins)")

        st.success("Schedule generated and saved.")
