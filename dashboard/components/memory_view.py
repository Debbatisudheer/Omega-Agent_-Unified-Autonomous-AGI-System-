# components/memory_view.py
import streamlit as st

def render_memory_view(memory):
    st.header("Memory Explorer")

    if not memory:
        st.warning("Memory backend not initialized.")
        return

    query = st.text_input("Search memory:", "reinforcement learning")
    top_k = st.slider("Top K", 1, 10, 3)

    if st.button("Search"):
        try:
            results = memory.query(query, top_k=top_k)
            for r in results:
                st.write("**Text:**")
                st.code(r.get("text"))
                st.write("**Metadata:**")
                st.write(r.get("metadata"))
                st.markdown("---")
        except Exception as e:
            st.error(str(e))
