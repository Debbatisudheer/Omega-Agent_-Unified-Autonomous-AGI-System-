# memory/vector_memory.py
"""
Unified VectorMemory for Omega AGI System.

Supports:
- Local JSON Memory (default)
- ChromaDB
- Pinecone (new 2024+ SDK)

Auto-selects backend using .env:
MEMORY_BACKEND=local|chroma|pinecone
"""

import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load embedding model
# ---------------------------
try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMB_MODEL = None
    print("⚠ No sentence-transformers. Using random fallback embeddings.")


def embed_text(text: str):
    """Generate vector embedding."""
    if EMB_MODEL:
        return EMB_MODEL.encode([text])[0]

    # fallback deterministic embedding
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    return rng.randn(384)


# ==============================
# BACKEND SELECTION
# ==============================
MEMORY_BACKEND = os.environ.get("MEMORY_BACKEND", "local").lower()
STORE_FILE = "memory_store.json"

CHROMA_COLLECTION = None
PINECONE_INDEX = None


# ---------------------------
# CHROMA BACKEND
# ---------------------------
if MEMORY_BACKEND == "chroma":
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_db"
        ))

        try:
            CHROMA_COLLECTION = client.get_collection("omega_memory")
        except:
            CHROMA_COLLECTION = client.create_collection("omega_memory")

        print("🔵 Using Chroma memory backend.")

    except Exception as e:
        print("❌ Failed to load Chroma. Falling back to local.", e)
        MEMORY_BACKEND = "local"


# ---------------------------
# PINECONE BACKEND (NEW SDK)
# ---------------------------
if MEMORY_BACKEND == "pinecone":
    try:
        from pinecone import Pinecone, ServerlessSpec

        api_key = os.environ["PINECONE_API_KEY"]
        index_name = os.environ.get("PINECONE_INDEX", "omega-memory")

        pc = Pinecone(api_key=api_key)

        # create index if missing
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        PINECONE_INDEX = pc.Index(index_name)

        print("🟢 Using Pinecone memory backend.")

    except Exception as e:
        print("❌ Pinecone init failed, falling back to local:", e)
        MEMORY_BACKEND = "local"


# ==============================
# MAIN CLASS
# ==============================
class VectorMemory:
    def __init__(self):
        self.backend = MEMORY_BACKEND

        if self.backend == "local":
            if os.path.exists(STORE_FILE):
                with open(STORE_FILE, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
            else:
                self.store = []

    # ======================================================
    # ADD MEMORY ENTRY
    # ======================================================
    def add(self, text, metadata=None):
        metadata = metadata or {}
        vec = embed_text(text).tolist()

        item = {
            "id": f"m_{int(datetime.utcnow().timestamp()*1000)}",
            "text": text,
            "vec": vec,
            "metadata": metadata,
            "time": datetime.utcnow().isoformat()
        }

        # ---------- LOCAL ----------
        if self.backend == "local":
            self.store.append(item)
            self._persist()
            return item

        # ---------- CHROMA ----------
        if self.backend == "chroma":
            CHROMA_COLLECTION.add(
                ids=[item["id"]],
                documents=[item["text"]],
                metadatas=[metadata],
                embeddings=[vec]
            )
            return item

        # ---------- PINECONE ----------
        if self.backend == "pinecone":
            PINECONE_INDEX.upsert([
                (item["id"], vec, metadata)
            ])
            return item

        return item

    # ======================================================
    # QUERY MEMORY
    # ======================================================
    def query(self, text, top_k=3):
        qv = embed_text(text)

        # ---------- LOCAL ----------
        if self.backend == "local":
            if not self.store:
                return []
            mats = np.array([np.array(i["vec"]) for i in self.store])
            sims = cosine_similarity([qv], mats)[0]
            idx = sims.argsort()[::-1][:top_k]
            return [self.store[i] for i in idx]

        # ---------- CHROMA ----------
        if self.backend == "chroma":
            res = CHROMA_COLLECTION.query(
                query_embeddings=[qv.tolist()],
                n_results=top_k,
                include=["documents", "metadatas"]
            )
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            return [{"text": docs[i], "metadata": metas[i]} for i in range(len(docs))]

        # ---------- PINECONE ----------
        if self.backend == "pinecone":
            res = PINECONE_INDEX.query(
                vector=qv.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            return [
                {
                    "metadata": match.get("metadata", {}),
                    "score": match.get("score")
                }
                for match in res["matches"]
            ]

        return []

    # ======================================================
    # LOCAL STORAGE SAVE
    # ======================================================
    def _persist(self):
        with open(STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2)
