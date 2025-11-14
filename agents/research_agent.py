# agents/research_agent.py

import requests
import datetime
import xml.etree.ElementTree as ET
from core.llm import complete
from memory.vector_memory import VectorMemory


class ResearchAgent:
    def __init__(self, memory: VectorMemory = None):
        self.memory = memory

    # ----------------------------------------------------
    # STEP 1: Fetch papers from arXiv API
    # ----------------------------------------------------
    def fetch_arxiv(self, query="reinforcement learning", max_results=2):
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query=all:{query}&start=0&max_results={max_results}"
        )

        try:
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                print("Error fetching arXiv:", res.text)
                return []
        except Exception as e:
            print("arXiv request failed:", e)
            return []

        root = ET.fromstring(res.text)

        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text

            papers.append({
                "title": title.strip(),
                "summary": summary.strip()
            })

        return papers

    # ----------------------------------------------------
    # STEP 2: Store papers in Pinecone vector memory
    # ----------------------------------------------------
    def store_papers(self, papers):
        for p in papers:
            text = p["title"] + "\n" + p["summary"]
            if self.memory:
                self.memory.add(text, metadata={"type": "research_paper"})

    # ----------------------------------------------------
    # STEP 3: Retrieve related memory (RAG)
    # ----------------------------------------------------
    def retrieve_related(self, query="reinforcement learning"):
        if not self.memory:
            return []
        return self.memory.query(query, top_k=3)

    # ----------------------------------------------------
    # STEP 4: Generate LLM summary + new experiment idea
    # ----------------------------------------------------
    def generate_summary(self, papers, retrieved):
        paper_text = "\n\n".join([
            f"TITLE: {p['title']}\nSUMMARY: {p['summary']}"
            for p in papers
        ])

        related_text = "\n".join([
            item.get("text", "")
            for item in retrieved
            if isinstance(item, dict)
        ])

        prompt = f"""
You are an AI Research Scientist.

Summarize the new research papers and propose ONE new experiment idea.

New Papers:
{paper_text}

Related Old Knowledge:
{related_text}

Give:
1) A very clear summary of the new findings.
2) One experiment idea to extend this work.
"""

        return complete(prompt)

    # ----------------------------------------------------
    # MAIN RESEARCH CYCLE
    # ----------------------------------------------------
    def run_cycle(self):
        # 1. Fetch real papers
        papers = self.fetch_arxiv("reinforcement learning", max_results=2)

        # 2. Store in memory
        if self.memory:
            self.store_papers(papers)

        # 3. Retrieve similar past memory from Pinecone
        retrieved = self.retrieve_related("reinforcement learning")

        # 4. LLM summarization + experiment proposal
        summary = self.generate_summary(papers, retrieved)

        # 5. Store summary
        if self.memory:
            self.memory.add(summary, metadata={"type": "research_summary"})

        # 6. Return output
        return {
            "time": datetime.datetime.utcnow().isoformat(),
            "papers": papers,
            "related_memory_count": len(retrieved),
            "summary": summary
        }


# --------------------------------------------------------
# WRAPPER FUNCTION REQUIRED BY omega.py
# --------------------------------------------------------
def run_research_agent(memory=None):
    agent = ResearchAgent(memory=memory)
    return agent.run_cycle()
