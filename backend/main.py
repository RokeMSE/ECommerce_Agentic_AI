import os
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text, Column, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector
from PIL import Image
import io
import torch
import clip
import numpy as np
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_DIM = 768 # For CLIP ViT-L/14

# --- Database Setup ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Review(Base):
    __tablename__ = "reviews"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=True)
    multimodal_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- Model Loading ---
class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on device: {self.device}")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)

    @torch.no_grad()
    def get_embeddings(self, text: str, image: Optional[Image.Image] = None) -> np.ndarray:
        text_tokens = clip.tokenize([text]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).cpu().numpy()
        
        if image:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input).cpu().numpy()
            # Weighted average fusion
            combined = 0.6 * text_features + 0.4 * image_features
            return combined / np.linalg.norm(combined)
        
        return text_features / np.linalg.norm(text_features)

model_manager = ModelManager()
app = FastAPI(title="Downscaled Multimodal API")

@app.on_event("startup")
def on_startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

# --- API ---
class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    similar_reviews: List[Dict[str, Any]]

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(text: str, image: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    pil_image = None
    if image:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

    # 1. Get embedding for the input
    query_embedding = model_manager.get_embeddings(text, pil_image)

    # 2. Find similar reviews in PostgreSQL
    similar = db.query(Review.id, Review.text, Review.multimodal_embedding.l2_distance(query_embedding).label('distance')) \
                .order_by(text('distance asc')) \
                .limit(5) \
                .all()
    
    similar_reviews = [{"id": str(r.id), "text": r.text, "score": 1 - r.distance} for r in similar]

    # 3. Sentiment Analysis (Placeholder - to be replaced by trained model)
    # This is where you would load and call your fine-tuned sentiment model
    sentiment = "negative" if "terrible" in text.lower() else "positive"

    return AnalyzeResponse(
        sentiment=sentiment,
        confidence=0.9,
        similar_reviews=similar_reviews
    )