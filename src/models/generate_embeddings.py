import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import clip
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        print("Loading text model...")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✓ Models loaded successfully")

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()  # 512-d
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return np.zeros(512, dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.text_model.encode(text, convert_to_numpy=True)  # 384-d for MiniLM

    def encode_multimodal(self, image_path: str, text: str) -> np.ndarray:
        img_emb = self.encode_image(image_path).astype(np.float32)
        txt_emb = self.encode_text(text).astype(np.float32)

        img_norm = np.linalg.norm(img_emb) + 1e-8
        txt_norm = np.linalg.norm(txt_emb) + 1e-8
        img_emb = img_emb / img_norm
        txt_emb = txt_emb / txt_norm

        combined = np.concatenate([img_emb, txt_emb])  # 512 + 384 = 896
        return combined


def generate_all_embeddings():
    df = pd.read_parquet("data/processed/products_clean.parquet")
    print(f"Processing {len(df)} products...")

    embedder = LocalEmbedder()

    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        text = f"{row['title']}. {row['description']}"
        image_path = row["image_path"]
        embeddings.append(embedder.encode_multimodal(image_path, text))

    df["embedding"] = embeddings

    os.makedirs("data/embeddings", exist_ok=True)
    df.to_pickle("data/embeddings/products_with_embeddings.pkl")
    print("✓ Saved embeddings to data/embeddings/products_with_embeddings.pkl")
    print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")

    return df


if __name__ == "__main__":
    generate_all_embeddings()
