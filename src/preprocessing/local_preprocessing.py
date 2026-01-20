import pandas as pd
import numpy as np
import os

print("Starting preprocessing...")

# Create output directory
os.makedirs("data/processed", exist_ok=True)

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
df.to_parquet("data/processed/products_clean.parquet", index=False)

