from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
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
import logging
import os
from hybrid_dense_parse_retrieval_approach import HybridRetrievalSystem 
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Sentiment Analysis API", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('inference_request_duration_seconds', 'Request latency')

class ModelManager:
    """Manages all ML models and connections"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
        
        # Initialize retrieval system
        self._initialize_retrieval()

        # Initialize cache
        try:
            self.cache = Redis(host='redis', port=6379, db=0, decode_responses=True)
            self.cache.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without cache.")
            self.cache = None
    
    def _load_models(self):
        """Load all required models"""
        try:
            # Load CLIP for embeddings - Use ViT-L/14 for 768-dim vectors
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            logger.info("CLIP model loaded successfully")
            
            # NOTE: In production, this would be the fine-tuned LLaVA model (cause the model is too big for local deployment)
            logger.info("Using demo sentiment classifier (placeholder for LLaVA)")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _initialize_retrieval(self):
        """Initialize the Hybrid Retrieval System"""
        try:
            retrieval_config = {
                'qdrant_host': os.getenv("QDRANT_HOST", "qdrant"),
                'qdrant_port': int(os.getenv("QDRANT_PORT", 6333)),
                'elasticsearch_url': os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200"),
                'es_user': os.getenv("ELASTICSEARCH_USER"),
                'es_password': os.getenv("ELASTICSEARCH_PASSWORD")
            }
            self.retrieval_system = HybridRetrievalSystem(config=retrieval_config)
            logger.info("Hybrid Retrieval System initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing retrieval system: {e}")
            self.retrieval_system = None

    def get_cache_key(self, text: str, image_hash: Optional[str]) -> str:
        """Generate cache key for request"""
        key_str = f"{text}_{image_hash if image_hash else 'no_image'}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @torch.no_grad()
    def analyze_sentiment(
        self, 
        text: str, 
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from text and/or image.
        
        NOTE: This is a DEMO implementation using rule-based + CLIP.
        In production, this would use the fine-tuned LLaVA model.
        """
        try:
            # Simple rule-based sentiment for demo
            text_lower = text.lower()

            # Negative keywords
            negative_words = ['terrible', 'awful', 'bad', 'worst', 'hate', 
                            'horrible', 'poor', 'disappointed', 'waste', 'broken']
            # Positive keywords
            positive_words = ['excellent', 'amazing', 'great', 'best', 'love',
                            'wonderful', 'perfect', 'fantastic', 'awesome', 'exceeded']
            neg_count = sum(1 for word in negative_words if word in text_lower) 
            pos_count = sum(1 for word in positive_words if word in text_lower)
            
            # Base prediction on text
            # WHichever has more occurance -> more weight -> more confidence
            if pos_count > neg_count:
                sentiment = "positive"
                confidence = 0.7 + (pos_count * 0.05) 
            elif neg_count > pos_count:
                sentiment = "negative"
                confidence = 0.7 + (neg_count * 0.05)
            else:
                sentiment = "neutral"
                confidence = 0.6
            
            # Adjust based on image if provided
            modality = "text_only"
            if image is not None:
                modality = "multimodal"
                # Get image embedding and adjust confidence
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                # Increase confidence for multimodal
                confidence = min(0.95, confidence + 0.1)
            confidence = min(0.99, confidence)
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "modality": modality,
                "method": "demo_classifier"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    @torch.no_grad()
    def get_embeddings(
        self, 
        text: str, 
        image: Optional[Image.Image] = None
    ) -> np.ndarray:
        """Generate CLIP embeddings for text and/or image"""
        try:
            # Text embedding
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
            if image is not None:
                # Image embedding
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                
                # Combine embeddings (weighted average)
                combined = 0.6 * text_features + 0.4 * image_features
                combined = combined / combined.norm(dim=-1, keepdim=True)
                return combined.cpu().numpy()[0]
            else:
                # Text only
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy()[0]
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# Initialize model manager
try:
    model_manager = ModelManager()
except Exception as e:
    logger.critical(f"Failed to initialize models: {e}")
    model_manager = None

# PYDANTIC MODELS
class AnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    modality: str
    similar_reviews: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    device: str
    cache_connected: bool
    models_loaded: bool

# API ENDPOINTS
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_review(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Analyze sentiment of a review (text and/or image).
    
    Args:
        text: Review text
        image: Optional product image
    
    Returns:
        Sentiment analysis results
    """
    start_time = time.time()
    
    if not model_manager:
        REQUEST_COUNT.labels(endpoint='analyze', status='error').inc()
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not text or not text.strip():
        REQUEST_COUNT.labels(endpoint='analyze', status='error').inc()
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        # Process image if provided
        pil_image = None
        image_hash = None
        
        if image:
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_hash = hashlib.md5(image_data).hexdigest()
        
        # Check cache
        cache_key = model_manager.get_cache_key(text, image_hash)
        cached_result = None
        
        if model_manager.cache:
            try:
                cached_result = model_manager.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for key: {cache_key}")
                    result = json.loads(cached_result)
                    result['cached'] = True
                    result['processing_time_ms'] = (time.time() - start_time) * 1000
                    REQUEST_COUNT.labels(endpoint='analyze', status='success').inc()
                    return AnalysisResponse(**result)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Perform analysis
        sentiment_result = model_manager.analyze_sentiment(text, pil_image)
        
                # Get embeddings for similarity search
        embeddings = model_manager.get_embeddings(text, pil_image)

        # Use the retrieval system to find similar reviews
        similar_reviews = []
        if model_manager.retrieval_system:
            try:
                similar_reviews = model_manager.retrieval_system.hybrid_search(
                    query_text=text,
                    query_embedding=embeddings,
                    k=5 # Return top 5 similar reviews
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")
        processing_time = (time.time() - start_time) * 1000
        
        response_data = {
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
            "modality": sentiment_result["modality"],
            "similar_reviews": similar_reviews,
            "processing_time_ms": processing_time,
            "cached": False
        }
        
        # Cache the result
        if model_manager.cache:
            try:
                model_manager.cache.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps({
                        "sentiment": sentiment_result["sentiment"],
                        "confidence": sentiment_result["confidence"],
                        "modality": sentiment_result["modality"],
                        "similar_reviews": similar_reviews
                    })
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        REQUEST_COUNT.labels(endpoint='analyze', status='success').inc()
        return AnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        REQUEST_COUNT.labels(endpoint='analyze', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cache_connected = False
    if model_manager and model_manager.cache:
        try:
            cache_connected = model_manager.cache.ping()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model_manager else "unhealthy",
        device=model_manager.device if model_manager else "unknown",
        cache_connected=cache_connected,
        models_loaded=model_manager is not None
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Multimodal Sentiment Analysis",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

# STARTUP/SHUTDOWN EVENTS
@app.on_event("startup")
async def startup_event():
    """Actions on startup"""
    logger.info("Starting Multimodal Sentiment Analysis Service")
    if model_manager:
        logger.info(f"Models loaded on {model_manager.device}")
        logger.info(f"Cache: {'Connected' if model_manager.cache else 'Disabled'}")
    else:
        logger.error("Models failed to load")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if model_manager and model_manager.cache:
        try:
            model_manager.cache.close()
        except:
            pass