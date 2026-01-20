from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI()

# Load models
embedder = MultimodalEmbedder()
ranking_model = RankingModel()
ranking_model.load_state_dict(torch.load('/mnt/models/ranking_model.pth'))
ranking_model.eval()

class RecommendationRequest(BaseModel):
    user_id: str
    query_text: str = None
    query_image_url: str = None
    top_k: int = 10

class RecommendationResponse(BaseModel):
    product_id: str
    score: float
    title: str
    price: float
    image_url: str

@app.post("/recommendations", response_model=list[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    try:
        # Get user features from database
        user_features = get_user_features(request.user_id)
        
        # Generate query embedding
        if request.query_text and request.query_image_url:
            image_bytes = download_image(request.query_image_url)
            query_emb = embedder.encode_multimodal(image_bytes, request.query_text)
        elif request.query_text:
            query_emb = embedder.encode_text(request.query_text)
        elif request.query_image_url:
            image_bytes = download_image(request.query_image_url)
            query_emb = embedder.encode_image(image_bytes)
        else:
            # Use user profile for cold start
            query_emb = get_user_profile_embedding(request.user_id)
        
        # Retrieve candidates
        candidates = get_recommendations(query_emb, user_features, top_k=request.top_k * 3)
        
        # Re-rank with personalization
        ranked_products = []
        for candidate in candidates:
            product_emb = torch.tensor(candidate['values'], dtype=torch.float32).unsqueeze(0)
            user_feat_tensor = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                score = ranking_model(product_emb, user_feat_tensor).item()
            
            ranked_products.append({
                'product_id': candidate['id'],
                'score': score,
                'title': candidate['metadata']['title'],
                'price': candidate['metadata']['price'],
                'image_url': f"/images/{candidate['id']}.jpg"
            })
        
        # Sort and return top-k
        ranked_products.sort(key=lambda x: x['score'], reverse=True)
        return ranked_products[:request.top_k]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}