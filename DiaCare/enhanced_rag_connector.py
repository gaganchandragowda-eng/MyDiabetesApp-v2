# enhanced_rag_connector.py
import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "diabetes-kb"
DIMENSION = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def enhanced_rag():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return {"status": "inactive", "reason": "PINECONE_API_KEY not set"}

    try:
        pc = Pinecone(api_key=api_key)
        index_names = [idx.name for idx in pc.list_indexes()]

        if INDEX_NAME not in index_names:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created index: {INDEX_NAME}")

        index = pc.Index(INDEX_NAME)
        embedder = SentenceTransformer(MODEL_NAME)

        kb_path = os.path.join("diabetes_kb_data", "processed", "sections.json")
        if not os.path.exists(kb_path):
            return {"status": "inactive", "reason": f"KB not found: {kb_path}"}

        with open(kb_path, "r", encoding="utf-8") as f:
            sections = json.load(f)

        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            print("Upserting KB...")
            vectors = []
            for i, text in enumerate(sections):
                emb = embedder.encode(text).tolist()
                vectors.append((str(i), emb, {"text": text}))
            index.upsert(vectors=vectors)
            print(f"Upserted {len(vectors)} vectors")

        def retrieve(query: str, k: int = 5):
            q_emb = embedder.encode(query).tolist()
            res = index.query(vector=q_emb, top_k=k, include_metadata=True)
            return [m["metadata"]["text"] for m in res.matches]

        return {"status": "active", "retrieve_fn": retrieve}

    except Exception as e:
        return {"status": "inactive", "reason": str(e)}