import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# ---------- STEP 0: SET BASE PATH ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "women_laws.json")
INDEX_PATH = os.path.join(BASE_DIR, "vector_db", "faiss_index.index")

# ---------- STEP 1: LOAD LAWS ----------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    laws = json.load(f)

# ---------- STEP 2: CREATE RICH TEXT FOR EMBEDDING ----------
def law_to_text(law):
    return (
        f"{law.get('law_name', '')}. "
        f"{law.get('category', '')}. "
        f"{law.get('description', '')}. "
        f"{' '.join(law.get('keywords', []))}. "
        f"{law.get('why_applicable', '')}"
    )

texts = [law_to_text(law) for law in laws]

print(f"Loaded {len(texts)} laws for embedding")

# ---------- STEP 3: LOAD EMBEDDING MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- STEP 4: CREATE EMBEDDINGS ----------
embeddings = model.encode(texts, convert_to_numpy=True)

dimension = embeddings.shape[1]

# ---------- STEP 5: CREATE FAISS INDEX ----------
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---------- STEP 6: SAVE INDEX ----------
faiss.write_index(index, INDEX_PATH)

print("✅ FAISS vector index rebuilt successfully!")
