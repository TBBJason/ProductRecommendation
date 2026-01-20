import sqlite3
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings

import torch
import clip
from sentence_transformers import SentenceTransformer



app = FastAPI(title="Local Recommendation API")

# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
text_model = SentenceTransformer("all-MiniLM-L6-v2")


def text_to_multimodal_query_embedding(text: str) -> list[float]:
    """
    Chroma collection stores 896-d vectors: [CLIP_image(512), MiniLM_text(384)].
    For a text-only query, use zeros for image part + normalized text embedding.
    """
    txt = text_model.encode(text, convert_to_numpy=True).astype(np.float32)
    txt = txt / (np.linalg.norm(txt) + 1e-8)
    img_zeros = np.zeros(512, dtype=np.float32)
    emb = np.concatenate([img_zeros, txt])  # 896-d
    return emb.tolist()


# Chroma
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection("products")


def get_db():
    return sqlite3.connect("data/recommendations.db")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class RecommendationResponse(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    score: float


@app.get("/")
def root():
    return {"message": "Local Recommendation API", "status": "running"}


@app.post("/search", response_model=list[RecommendationResponse])
def search_products(request: SearchRequest):
    query_embedding = text_to_multimodal_query_embedding(request.query)

    results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)

    recommendations = []
    for id_, metadata, distance in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        recommendations.append(
            {
                "product_id": id_,
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "price": float(metadata.get("price", 0.0) or 0.0),
                "score": float(1.0 / (1.0 + distance)),
            }
        )

    return recommendations


@app.get("/product/{product_id}")
def get_product(product_id: str):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Product not found")

    return {
        "product_id": row[0],
        "title": row[1],
        "description": row[2],
        "category": row[3],
        "brand": row[4],
        "price": row[5],
        "rating": row[6],
        "num_reviews": row[7],
    }


@app.get("/recommendations/{user_id}")
def get_user_recommendations(user_id: str, top_k: int = 10):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT p.category, COUNT(*) as count
        FROM interactions i
        JOIN products p ON i.product_id = p.product_id
        WHERE i.user_id = ?
        GROUP BY p.category
        ORDER BY count DESC
        LIMIT 3
        """,
        (user_id,),
    )

    preferred_categories = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not preferred_categories:
        return {"message": "No history for user", "recommendations": []}

    # Use a text query representing the user's category interests + filter to those categories.
    category_query = " ".join(preferred_categories)
    query_embedding = text_to_multimodal_query_embedding(category_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"category": {"$in": preferred_categories}},
    )

    recs = []
    for id_, metadata, distance in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        recs.append(
            {
                "product_id": id_,
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "price": float(metadata.get("price", 0.0) or 0.0),
                "score": float(1.0 / (1.0 + distance)),
            }
        )

    return {"user_id": user_id, "preferred_categories": preferred_categories, "recommendations": recs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000