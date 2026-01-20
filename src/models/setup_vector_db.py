import os
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

os.makedirs("data/chroma_db", exist_ok=True)

client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma_db")
)

collection = client.get_or_create_collection(
    name="products",
    metadata={"description": "Product embeddings for recommendations"},
)

print("✓ ChromaDB initialized")

df = pd.read_pickle("data/embeddings/products_with_embeddings.pkl")
print(f"Loaded {len(df)} products with embeddings")

batch_size = 100
for i in tqdm(range(0, len(df), batch_size), desc="Indexing products"):
    batch = df.iloc[i : i + batch_size]

    ids = batch["product_id"].tolist()
    embeddings = [e.tolist() for e in batch["embedding"].tolist()]
    metadatas = batch[["title", "category", "brand", "price"]].to_dict("records")
    documents = batch["description"].tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

print(f"✓ Indexed {collection.count()} products in ChromaDB")

# Simple sanity check: query with the first product embedding
test_embedding = df.iloc[0]["embedding"].tolist()
results = collection.query(query_embeddings=[test_embedding], n_results=5)

print("\nTest query results:")
for rank, (id_, distance) in enumerate(
    zip(results["ids"][0], results["distances"][0]), start=1
):
    print(f"{rank}. Product {id_} (distance: {distance:.4f})")

print("\n✓ Vector database ready!")
