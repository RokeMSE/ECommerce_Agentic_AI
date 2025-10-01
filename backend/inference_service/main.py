from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import clip
from PIL import Image
import io
import numpy as np
from redis import Redis
import hashlib
import json
import time

app = FastAPI(title="Multimodal Sentiment Analysis API", version="2.0")

# Global model loading
class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")    
        
        print("Loading VLM model for sentiment analysis...")
        
        # NOTE: LLaVA is too large of a model so for local I'm just putting it as a placeholder.
        # self.vlm_model = LlavaForConditionalGeneration.from_pretrained(...)
        # self.vlm_processor = AutoProcessor.from_pretrained(...)
        
        print("Loading CLIP model for embeddings...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        
        print("Connecting to Redis cache...")
        self.cache = Redis(host='redis', port=6379, db=0)
    
    def get_cache_key(self, text: str, image_hash: Optional[str]) -> str:
        """ 
        text (str): The text to generate a key for.
        image_hash (Optional[str]): The hash of the image to generate a key for. 
        """
        key = f"{text}_{image_hash}" if image_hash else text
        return hashlib.md5(key.encode()).hexdigest() # A hexadecimal string representing the cache key.

    @torch.no_grad()
    def analyze_sentiment(self, text: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        # NOTE: THIS IS ONLY APLALCEHOLDER FOR LLaVA INFERENCE
        print("Analyzing sentiment (placeholder logic)...")
        sentiment = "negative" if "terrible" in text.lower() else "positive"
        return {"sentiment": sentiment, "confidence": 0.9, "modality": "multimodal" if image else "text_only"}

    @torch.no_grad()
    def get_embeddings(self, text: str, image: Optional[Image.Image] = None) -> np.ndarray:
        text_tokens = clip.tokenize([text]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).cpu().numpy()
        
        if image:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input).cpu().numpy()
            combined = 0.6 * text_features + 0.4 * image_features
            return combined / np.linalg.norm(combined)
        
        return text_features / np.linalg.norm(text_features)

model_manager = ModelManager()

# Pydantic Models
class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    return_similar: bool = Field(default=True)
    k: int = Field(default=5)

class AnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    modality: str
    similar_reviews: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float

# API Endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_review(text: str, image: Optional[UploadFile] = File(None)):
    start_time = time.time()
    pil_image = None
    if image:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    sentiment_result = model_manager.analyze_sentiment(text, pil_image)
    embeddings = model_manager.get_embeddings(text, pil_image)
    
    # Placeholder for hybrid retrieval call
    similar_reviews = [] # await retrieve_similar(embeddings, k)

    processing_time = (time.time() - start_time) * 1000
    
    return AnalysisResponse(
        sentiment=sentiment_result["sentiment"],
        confidence=sentiment_result["confidence"],
        modality=sentiment_result["modality"],
        similar_reviews=similar_reviews,
        processing_time_ms=processing_time
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": model_manager.device, "cache_connected": model_manager.cache.ping()}