import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import subprocess
import chromadb
from sentence_transformers import SentenceTransformer
import plotly.express as px

st.set_page_config(page_title="Product Recommendations", layout="wide")

# ============================================================================
# HELPER: Run scripts with correct Python interpreter
# ============================================================================

def run_script(script_path):
    """Run a Python script using the current interpreter (ensures packages are found)."""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"Script failed:\n{result.stderr}")
        return True
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Script timed out after 10 minutes: {script_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to run {script_path}: {str(e)}")

# ============================================================================
# CHECK IF PIPELINE IS COMPLETE & INITIALIZE IF NEEDED
# ============================================================================

required_files = [
    "data/chroma_db",
    "data/recommendations.db",
]

pipeline_complete = all(os.path.exists(f) for f in required_files)

if not pipeline_complete:
    st.warning("⏳ First load: Initializing pipeline...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        steps = [
            ("Generating sample data...", "scripts/generate_sample_data.py", 0.2),
            ("Preprocessing data...", "src/preprocessing/local_preprocessing.py", 0.4),
            ("Creating embeddings (this may take a few minutes)...", "src/models/generate_embeddings.py", 0.7),
            ("Indexing vector database...", "src/models/setup_vector_db.py", 0.85),
            ("Setting up database...", "src/database/init_db.py", 1.0),
        ]
        
        for step_name, script_path, progress_value in steps:
            status_text.text(step_name)
            run_script(script_path)
            progress_bar.progress(progress_value)
        
        status_text.text("✅ Pipeline complete!")
        progress_bar.progress(1.0)
        st.success("Data is ready! Refresh the page to start using the app.")
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.info("Please check the error above and try again, or refresh the page.")
        st.stop()

# ============================================================================
# LOAD MODELS & DATABASE (only if pipeline is complete)
# ============================================================================

@st.cache_resource
def get_connection():
    return sqlite3.connect("data/recommendations.db", check_same_thread=False)

@st.cache_resource
def load_models():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    collection = chroma_client.get_collection("products")
    return text_model, collection

conn = get_connection()
text_model, collection = load_models()

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🛍️ Product Recommendation System")
st.markdown("Multimodal product search + analytics")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Search", "Analytics", "System Status"])

# ============================================================================
# PAGE: SEARCH
# ============================================================================
if page == "Search":
    st.header("Product Search")
    st.markdown("Search for products using text descriptions")
    
    query = st.text_input(
        "Search for products:",
        placeholder="e.g., 'blue electronics', 'leather wallet', 'outdoor gear'",
        key="search_query"
    )
    top_k = st.slider("Number of results:", 5, 20, 10)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                import numpy as np
                
                # Generate text embedding (384-dim)
                text_embedding = text_model.encode(query)
                
                # Add zero image embedding (512-dim) for consistency with stored embeddings
                image_embedding = np.zeros(512)
                
                # Combine to 896-dim (matches ChromaDB collection)
                query_embedding = np.concatenate([text_embedding, image_embedding]).tolist()
                
                # Search in ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                if results["ids"] and results["ids"][0]:
                    st.subheader(f"Found {len(results['ids'][0])} results")
                    
                    for i, (doc_id, metadata, distance) in enumerate(
                        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
                    ):
                        # Calculate similarity score (inverse of distance)
                        score = 1 - (distance / 2)  # Normalize to 0-1
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {i+1}. {metadata.get('title', 'N/A')}")
                            st.write(f"**Category:** {metadata.get('category', 'N/A')}")
                            st.write(f"**Similarity Score:** {score:.1%}")
                        
                        with col2:
                            st.metric("Price", f"${float(metadata.get('price', 0)):.2f}")
                        
                        with col3:
                            st.metric("ID", metadata.get('product_id', 'N/A'))
                        
                        st.divider()
                else:
                    st.warning("No results found. Try a different search query.")
            
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.info("Please check the System Status page to ensure the database is ready.")


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "Analytics":
    st.header("System Analytics")
    st.markdown("Product inventory and interaction statistics")
    
    try:
        # Load data
        df_products = pd.read_sql("SELECT * FROM products", conn)
        df_interactions = pd.read_sql("SELECT * FROM interactions", conn)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", len(df_products))
        col2.metric("Categories", int(df_products["category"].nunique()))
        col3.metric("Avg Price", f"${df_products['price'].mean():.2f}")
        col4.metric("Total Interactions", len(df_interactions))
        
        st.divider()
        
        # Products by category
        st.subheader("Products by Category")
        category_counts = df_products["category"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig_category = px.bar(
            category_counts,
            x="category",
            y="count",
            labels={"category": "Category", "count": "Count"},
            title="Product Distribution by Category",
            color="count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Price distribution
        st.subheader("Price Distribution")
        fig_price = px.histogram(
            df_products,
            x="price",
            nbins=30,
            title="Product Price Distribution",
            labels={"price": "Price ($)"},
            color_discrete_sequence=["#00A86B"]
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Interaction types
        st.subheader("Interaction Types")
        interaction_counts = df_interactions["interaction_type"].value_counts()
        fig_interaction = px.pie(
            values=interaction_counts.values,
            names=interaction_counts.index,
            title="Distribution of Interaction Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_interaction, use_container_width=True)
        
        # Recent interactions
        st.subheader("Recent Interactions")
        df_interactions_recent = pd.read_sql(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 10",
            conn
        )
        st.dataframe(df_interactions_recent, use_container_width=True)
    
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")

# ============================================================================
# PAGE: SYSTEM STATUS
# ============================================================================

else:  # System Status
    st.header("System Status")
    
    # Data files
    st.subheader("📁 Data Files")
    files_to_check = [
        ("Products Data", "data/processed/products_clean.parquet"),
        ("Embeddings", "data/embeddings/products_with_embeddings.pkl"),
        ("SQLite Database", "data/recommendations.db"),
        ("Vector Index (ChromaDB)", "data/chroma_db"),
    ]
    
    all_ready = True
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✅ Ready" if exists else "❌ Missing"
        st.write(f"**{name}:** {status}")
        if not exists:
            all_ready = False
    
    st.divider()
    
    # Model information
    st.subheader("🧠 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Vision Model:**")
        st.code("CLIP ViT-B/32\n512-dimensional embeddings")
    
    with col2:
        st.write("**Text Model:**")
        st.code("all-MiniLM-L6-v2\n384-dimensional embeddings")
    
    st.write("**Combined Embedding Dimension:** 896-dim")
    
    st.divider()
    
    # System resources
    st.subheader("💻 System Resources")
    try:
        import psutil
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        col1, col2 = st.columns(2)
        col1.metric("CPU Usage", f"{cpu_usage}%")
        col2.metric("RAM Usage", f"{ram_usage}%")
    except Exception as e:
        st.info(f"Could not retrieve system metrics: {e}")
    
    st.divider()
    
    # Status summary
    st.subheader("📊 Overall Status")
    if all_ready:
        st.success("✅ All systems ready! The app is fully functional.")
    else:
        st.warning("⏳ Pipeline is still initializing. Please refresh in a few minutes.")
