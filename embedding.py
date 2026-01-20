import torch
import clip
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import io

class MultimodalEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Vision encoder (CLIP)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Text encoder (can swap for OpenAI)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def encode_image(self, image_bytes):
        """Generate image embeddings"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return np.zeros(512)  # Return zero vector on error
    
    def encode_text(self, text):
        """Generate text embeddings"""
        try:
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error encoding text: {e}")
            return np.zeros(384)
    
    def encode_multimodal(self, image_bytes, text, alpha=0.7):
        """
        Combine image and text embeddings
        alpha: weight for image embedding (1-alpha for text)
        """
        img_emb = self.encode_image(image_bytes)
        txt_emb = self.encode_text(text)
        
        # Normalize embeddings
        img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
        txt_emb = txt_emb / (np.linalg.norm(txt_emb) + 1e-8)
        
        # Weighted combination
        combined = np.concatenate([
            alpha * img_emb,
            (1 - alpha) * txt_emb
        ])
        
        return combined

# PySpark UDF for batch embedding generation
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd

embedder = MultimodalEmbedder()

@pandas_udf(ArrayType(FloatType()))
def generate_embeddings_udf(image_series: pd.Series, text_series: pd.Series) -> pd.Series:
    embeddings = []
    for img_bytes, text in zip(image_series, text_series):
        emb = embedder.encode_multimodal(img_bytes, text)
        embeddings.append(emb.tolist())
    return pd.Series(embeddings)

# Apply to DataFrame
df_with_embeddings = df_final.withColumn(
    "embedding",
    generate_embeddings_udf(col("image_preprocessed"), col("description_clean"))
)

df_with_embeddings.write.format("delta").mode("overwrite") \
    .save("/mnt/gold/products_with_embeddings")