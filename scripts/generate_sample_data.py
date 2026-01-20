import os
import random
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)
random.seed(42)


def generate_sample_products(n=1000):
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books", "Toys"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

    adjectives = ["Premium", "Deluxe", "Essential", "Classic", "Modern", "Vintage"]
    nouns = ["Widget", "Gadget", "Device", "Item", "Product", "Tool"]

    products = []
    for i in range(n):
        category = random.choice(categories)
        brand = random.choice(brands)

        title = f"{random.choice(adjectives)} {category} {random.choice(nouns)}"
        description = (
            f"High-quality {category.lower()} from {brand}. "
            f"Perfect for everyday use. Premium materials and craftsmanship."
        )

        products.append(
            {
                "product_id": f"P{i:05d}",
                "title": title,
                "description": description,
                "category": category,
                "brand": brand,
                "price": round(random.uniform(10, 500), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "num_reviews": random.randint(0, 1000),
            }
        )

    df = pd.DataFrame(products)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/products.csv", index=False)
    print(f"Generated {n} products -> data/raw/products.csv")
    return df


def generate_sample_images(df, img_dir="data/raw/images"):
    os.makedirs(img_dir, exist_ok=True)

    for idx, row in df.iterrows():
        color = tuple(np.random.randint(50, 200, 3))
        img = Image.new("RGB", (224, 224), color)
        img.save(f"{img_dir}/{row['product_id']}.jpg")

        if idx % 100 == 0:
            print(f"Generated {idx}/{len(df)} images")

    print(f"All images saved to {img_dir}/")


def generate_user_interactions(df, n_users=500, n_interactions=10000):
    interactions = []
    for _ in range(n_interactions):
        user_id = f"U{random.randint(0, n_users):05d}"
        product_id = df.sample(1)["product_id"].values[0]
        interaction_type = random.choices(
            ["view", "click", "add_to_cart", "purchase"],
            weights=[0.5, 0.3, 0.15, 0.05],
        )[0]

        interactions.append(
            {
                "user_id": user_id,
                "product_id": product_id,
                "interaction_type": interaction_type,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 90)),
            }
        )

    df_interactions = pd.DataFrame(interactions)
    df_interactions.to_csv("data/raw/interactions.csv", index=False)
    print(f"Generated {n_interactions} interactions -> data/raw/interactions.csv")
    return df_interactions


if __name__ == "__main__":
    print("Generating sample data...")

    df_products = generate_sample_products(n=1000)
    generate_sample_images(df_products)
    generate_user_interactions(df_products)

    print("\n✓ Sample data generation complete!")
