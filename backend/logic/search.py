import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load laws database
with open("data/women_laws.json", "r", encoding="utf-8") as f:
    laws = json.load(f)

# Load FAISS vector index
index = faiss.read_index("vector_db/faiss_index.index")


def find_law(user_problem, top_k=3):
    """
    Returns top_k most relevant laws for the user query
    """

    # Convert query to embedding
    query_vector = model.encode([user_problem])

    # Search in FAISS index
    distances, indices = index.search(np.array(query_vector), k=top_k)

    results = []

    for idx, score in zip(indices[0], distances[0]):
        law = laws[idx]

        results.append({
            "law_name": law.get("law_name"),
            "category": law.get("category"),
            "description": law.get("description"),
            "actions": law.get("actions"),
            "similarity_score": float(score)
        })

    return results