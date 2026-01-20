import chromadb
import pickle
import os

print("Loading embeddings...")
with open("data/embeddings/products_with_embeddings.pkl", "rb") as f:
    embeddings_data = pickle.load(f)

print(f"Setting up vector database with {len(embeddings_data)} products...")

# Create persistent client (new API)
client = chromadb.PersistentClient(path="data/chroma_db")

# Delete collection if it exists
try:
    client.delete_collection(name="products")
except:
    pass

# Create new collection
collection = client.create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"}
)

# Add embeddings to collection
for item in embeddings_data:
    collection.add(
        ids=[str(item["product_id"])],
        embeddings=[item["combined_embedding"].tolist()],
        metadatas=[{
            "title": item["title"],
            "category": item["category"],
            "price": float(item["price"]),
            "product_id": str(item["product_id"]),
        }],
        documents=[f"{item['title']} {item['category']}"]
    )

