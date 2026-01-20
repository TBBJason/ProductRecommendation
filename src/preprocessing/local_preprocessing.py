import pandas as pd
import numpy as np
import os
import shutil

print("Starting preprocessing...")

# Create output directory (remove if it exists as a file/bad state)
output_dir = "data/processed"
if os.path.exists(output_dir) and os.path.isfile(output_dir):
    os.remove(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load sample data
df = pd.read_csv("data/raw/products.csv")

print(f"Loaded {len(df)} products")

# Basic cleaning
df = df.dropna(subset=['title', 'category', 'price'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]

# Remove duplicates
df = df.drop_duplicates(subset=['title'])

print(f"Cleaned to {len(df)} products")

# Save cleaned data
output_file = os.path.join(output_dir, "products_clean.parquet")
df.to_parquet(output_file, index=False)

