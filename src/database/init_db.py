import os
import sqlite3
import pandas as pd

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect("data/recommendations.db")
cursor = conn.cursor()

print("Connected to SQLite database")

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    category TEXT,
    brand TEXT,
    price REAL,
    rating REAL,
    num_reviews INTEGER
)
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    product_id TEXT,
    interaction_type TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
"""
)

cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_product ON interactions(product_id)")

conn.commit()
print("Tables created")

df_products = pd.read_parquet("data/processed/products_clean.parquet")
df_products = df_products[
    ["product_id", "title", "description", "category", "brand", "price", "rating", "num_reviews"]
]
df_products.to_sql("products", conn, if_exists="replace", index=False)
print(f"✓ Inserted {len(df_products)} products")

df_interactions = pd.read_csv("data/raw/interactions.csv")
df_interactions.to_sql("interactions", conn, if_exists="replace", index=False)
print(f"✓ Inserted {len(df_interactions)} interactions")

unique_users = df_interactions["user_id"].unique()
df_users = pd.DataFrame({"user_id": unique_users})
df_users.to_sql("users", conn, if_exists="replace", index=False)
print(f"✓ Inserted {len(df_users)} users")

cursor.execute("SELECT COUNT(*) FROM products")
count = cursor.fetchone()[0]
print(f"\n✓ Database ready with {count} products")

conn.close()
