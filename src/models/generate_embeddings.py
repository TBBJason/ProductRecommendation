import pandas as pd
import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load cleaned products
df = pd.read_parquet("data/processed/products_clean.parquet")
print(f"Processing {len(df)} products...")

embeddings_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
    try:
        title = row.get("title", "")
        category = row.get("category", "")
        
        # Create text description
        text_desc = f"{title} {category}"
        
        # Generate text embedding
        text_embedding = text_model.encode(text_desc)
        
        # Try to load image, but if it doesn't exist or fails, use zeros
        image_embedding = np.zeros(512)  # CLIP ViT-B/32 uses 512-dim
        
        image_path = row.get("image_path")
        if image_path and os.path.exists(image_path):
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_embedding = clip_model.encode_image(image_tensor).cpu().numpy()[0]
            except Exception as e:
                print(f"Warning: Could not process image for {title}: {e}")
                image_embedding = np.zeros(512)
        
        # Combine embeddings (896-dim total)
        combined_embedding = np.concatenate([text_embedding, image_embedding])
        
        embeddings_data.append({
            "product_id": row.get("product_id", idx),
            "title": title,
            "category": category,
            "price": row.get("price", 0),
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "combined_embedding": combined_embedding,
        })
    
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

# Save embeddings
os.makedirs("data/embeddings", exist_ok=True)
with open("data/embeddings/products_with_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_data, f)

print(f"✓ Generated embeddings for {len(embeddings_data)} products")
