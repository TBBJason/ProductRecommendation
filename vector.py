from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")

# Create index
index_name = "product-recommendations"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=896,  
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

# Upsert embeddings from Delta table
df_embeddings = spark.read.format("delta").load("/mnt/gold/products_with_embeddings")

def upsert_to_pinecone(partition_data):
    """Upsert function for each partition"""
    vectors = []
    for row in partition_data:
        vectors.append({
            'id': str(row['product_id']),
            'values': row['embedding'],
            'metadata': {
                'title': row['title_clean'],
                'category': row['category'],
                'price': float(row['price']),
                'brand': row['brand']
            }
        })
    
    if vectors:
        index.upsert(vectors=vectors)
    
    return iter([])

df_embeddings.rdd.foreachPartition(upsert_to_pinecone)

# Query example
def get_recommendations(query_embedding, user_preferences, top_k=20):
    """
    Get candidate products from vector DB
    """
    # Initial retrieval
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k * 2,  # Over-fetch for re-ranking
        include_metadata=True,
        filter={
            "category": {"$in": user_preferences.get('categories', [])},
            "price": {"$lte": user_preferences.get('max_price', 10000)}
        }
    )
    
    return results['matches']