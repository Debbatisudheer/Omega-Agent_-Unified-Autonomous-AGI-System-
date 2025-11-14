from pinecone import Pinecone
import os

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

pc.delete_index("omega-memory")
print("Deleted omega-memory successfully.")
