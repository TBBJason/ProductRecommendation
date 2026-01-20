#!/bin/bash

# setup.sh - Initial setup script for recommendation system

set -e  # Exit on error

echo "Setting up Recommendation System..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED} Python $REQUIRED_VERSION or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/{raw,processed,embeddings,chroma_db,models}
mkdir -p data/raw/images
mkdir -p notebooks
mkdir -p logs
mkdir -p deployment/{docker,kubernetes,monitoring}
mkdir -p tests
echo -e "${GREEN}✓ Directories created${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create .env file from example
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///data/recommendations.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Vector Database
CHROMA_PERSIST_DIR=data/chroma_db

# Models
CLIP_MODEL_NAME=ViT-B/32
TEXT_MODEL_NAME=all-MiniLM-L6-v2
DEVICE=cpu

# Cache
ENABLE_CACHE=true
CACHE_TTL=3600
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Create .gitignore
echo ""
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
build/
dist/
*.egg-info/

# Data
data/raw/*
data/processed/*
data/embeddings/*
data/chroma_db/*
data/models/*
*.db
*.pkl
*.parquet

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/

# Models
models/
!models/.gitkeep
EOF
    echo -e "${GREEN}✓ .gitignore created${NC}"
else
    echo -e "${YELLOW}⚠ .gitignore already exists${NC}"
fi

# Generate sample data
echo ""
read -p "Generate sample data? (1000 products) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating sample data..."
    python scripts/generate_sample_data.py --num-products 1000
    echo -e "${GREEN}✓ Sample data generated${NC}"
fi

# Process data
echo ""
read -p "Process data and generate embeddings? (takes ~5-10 min) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Processing data..."
    python scripts/run_pipeline.py
    echo -e "${GREEN}✓ Data processing complete${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "2. Start the API server:"
echo "   ${YELLOW}uvicorn src.api.app:app --reload${NC}"
echo ""
echo "3. Start the dashboard (in another terminal):"
echo "   ${YELLOW}streamlit run src/dashboard/app.py${NC}"
echo ""
echo "4. Or use Docker:"
echo "   ${YELLOW}docker-compose up -d${NC}"
echo ""
echo "Access points:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Dashboard: http://localhost:8501"
echo ""
echo -e "${GREEN}Happy recommending! 🎉${NC}"
echo ""