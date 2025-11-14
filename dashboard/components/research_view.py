# components/research_view.py
import streamlit as st
from agents.research_agent import ResearchAgent

def render_research_view(memory):
    st.header("Research Lab")
    st.write("Fetch & summarize arXiv papers.")

    query = st.text_input("arXiv Query:", "reinforcement learning")
    n = st.number_input("Number of Papers:", 1, 10, 2)

    if st.button("Fetch + Summarize"):
        agent = ResearchAgent(memory=memory)
        papers = agent.fetch_arxiv(query=query, max_results=n)

        if not papers:
            st.warning("No papers fetched.")
        else:
            st.success("Fetched Papers:")
            for p in papers:
                st.subheader(p["title"])
                st.write(p["summary"])

            st.info("Summaries saved to memory.")
