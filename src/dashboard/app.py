import os
import sqlite3
import requests
import psutil

import pandas as pd
import streamlit as st
import plotly.express as px
import os
import subprocess
import time

st.set_page_config(page_title="Product Recommendations", layout="wide")

@st.cache_resource
def check_and_init_pipeline():
    """Check if pipeline is complete; if not, return False."""
    artifacts = [
        "data/processed/products_clean.parquet",
        "data/embeddings/products_with_embeddings.pkl",
        "data/chroma_db",
        "data/recommendations.db",
    ]
    return all(os.path.exists(a) for a in artifacts)

# Check status
pipeline_ready = check_and_init_pipeline()

if not pipeline_ready:
    st.warning("⏳ Pipeline initializing... (first load only, takes ~5 min)")
    st.info("""
    Your app is generating embeddings for the first time.
    This happens only once. Please check back in a few minutes.
    
    Steps:
    1. Generating sample data...
    2. Preprocessing...
    3. Creating embeddings (longest step)...
    4. Indexing vectors...
    5. Setting up database...
    """)
    
    # Trigger pipeline in background (don't wait)
    if not os.path.exists(".pipeline_started"):
        with open(".pipeline_started", "w") as f:
            f.write("1")
        
        # Start pipeline in background
        subprocess.Popen([
            "python", "scripts/generate_sample_data.py"
        ])
    
    st.stop()

# If we get here, pipeline is ready
st.title("🛍️ Product Recommendation System")
st.markdown("Local prototype - zero cloud costs!")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Search", "Analytics", "System Status"])


if page == "Search":
    st.header("Product Search")

    query = st.text_input("Search for products:", placeholder="e.g., 'blue electronics'")
    top_k = st.slider("Number of results:", 5, 20, 10)

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    "http://localhost:8000/search",
                    json={"query": query, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
                results = response.json()

                st.subheader(f"Found {len(results)} results")

                for result in results:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {result['title']}")
                        st.write(f"**Category:** {result['category']}")
                        st.write(f"**Score:** {result['score']:.3f}")
                    with col2:
                        st.metric("Price", f"${result['price']:.2f}")
                    st.divider()

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the API server is running: `python src/api/app.py`")


elif page == "Analytics":
    st.header("System Analytics")

    df_products = pd.read_sql("SELECT * FROM products", conn)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(df_products))
    col2.metric("Categories", int(df_products["category"].nunique()))
    col3.metric("Avg Price", f"${df_products['price'].mean():.2f}")

    st.subheader("Products by Category")
    category_counts = df_products["category"].value_counts()
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={"x": "Category", "y": "Count"},
        title="Product Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Distribution")
    fig = px.histogram(df_products, x="price", nbins=30, title="Product Prices")
    st.plotly_chart(fig, use_container_width=True)

    df_interactions = pd.read_sql("SELECT * FROM interactions", conn)

    st.subheader("Interaction Types")
    interaction_counts = df_interactions["interaction_type"].value_counts()
    fig = px.pie(
        values=interaction_counts.values,
        names=interaction_counts.index,
        title="Interaction Types",
    )
    st.plotly_chart(fig, use_container_width=True)


else:
    st.header("System Status")

    st.subheader("Data Files")
    files_to_check = [
        ("Products", "data/processed/products_clean.parquet"),
        ("Embeddings", "data/embeddings/products_with_embeddings.pkl"),
        ("Database", "data/recommendations.db"),
        ("ChromaDB", "data/chroma_db"),
    ]

    for name, path in files_to_check:
        exists = os.path.exists(path)
        st.write(f"**{name}:** {'✅ Ready' if exists else '❌ Missing'}")

    st.subheader("API Status")
    try:
        resp = requests.get("http://localhost:8000/", timeout=5)
        if resp.status_code == 200:
            st.success("API is running")
        else:
            st.error(f"❌ API returned status {resp.status_code}")
    except Exception:
        st.warning("⚠️ API is not running. Start it with: `python src/api/app.py`")

    st.subheader("Model Information")
    st.write("**Vision Model:** CLIP ViT-B/32 (512-dim)")
    st.write("**Text Model:** all-MiniLM-L6-v2 (384-dim)")
    st.write("**Combined Embedding:** 896-dim")

    st.subheader("System Resources")
    st.write(f"**CPU Usage:** {psutil.cpu_percent()}%")
    st.write(f"**RAM Usage:** {psutil.virtual_memory().percent}%")
