from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Optional
import torch

class HybridRetrievalSystem:
    """
    Advanced retrieval combining:
    1. Dense retrieval (CLIP embeddings in Qdrant)
    2. Sparse retrieval (BM25 in Elasticsearch)
    3. Cross-encoder reranking for final precision
    """
    def __init__(self, config: Dict):
        # Qdrant for dense vector search
        self.qdrant = QdrantClient(
            host=config['qdrant_host'],
            port=config['qdrant_port']
        )
        
        # Elasticsearch for sparse (keyword) search
        self.es = Elasticsearch(
            [config['elasticsearch_url']],
            basic_auth=(config['es_user'], config['es_password'])
        )
        
        # Cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.collection_name = "multimodal_reviews"
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize Qdrant collection and Elasticsearch index"""
        
        # Qdrant collection (768-dim CLIP embeddings)
        try:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # CLIP ViT-L/14 dimension
                    distance=Distance.COSINE
                )
            )
        except:
            pass  # Collection already exists
        
        # Elasticsearch index
        es_mapping = {
            "mappings": {
                "properties": {
                    "review_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "english"},
                    "sentiment": {"type": "keyword"},
                    "product_name": {"type": "text"},
                    "rating": {"type": "float"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        try:
            self.es.indices.create(
                index=self.collection_name,
                body=es_mapping
            )
        except:
            pass  # Index already exists
    
    def index_review(
        self,
        review_id: str,
        text: str,
        embedding: np.ndarray,
        metadata: Dict
    ):
        """
        Index a review in both Qdrant (dense) and Elasticsearch (sparse)
        """
        
        # Index in Qdrant (vector search)
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=review_id,
                    vector=embedding.tolist(),
                    payload=metadata
                )
            ]
        )
        
        # Index in Elasticsearch (keyword search)
        doc = {
            "review_id": review_id,
            "text": text,
            **metadata
        }
        
        self.es.index(
            index=self.collection_name,
            id=review_id,
            document=doc
        )
    
    def dense_search(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Dense vector search using CLIP embeddings.
        Captures semantic similarity.
        """
        
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            query_filter=filter_conditions
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                **hit.payload
            }
            for hit in search_result
        ]
    
    def sparse_search(
        self,
        query_text: str,
        k: int = 20,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Sparse keyword search using BM25.
        Captures exact term matches.
        """
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query_text}}
                    ]
                }
            },
            "size": k
        }
        
        if filter_conditions:
            query_body["query"]["bool"]["filter"] = [
                {"term": {k: v}} for k, v in filter_conditions.items()
            ]
        
        response = self.es.search(
            index=self.collection_name,
            body=query_body
        )
        
        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            for hit in response["hits"]["hits"]
        ]
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 10,
        alpha: float = 0.5,  # Weight between dense and sparse
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Combine dense and sparse search with reciprocal rank fusion.
        
        Args:
            alpha: 0 = pure sparse, 1 = pure dense, 0.5 = balanced
        """
        
        # Get candidates from both methods
        dense_results = self.dense_search(query_embedding, k=k*2)
        sparse_results = self.sparse_search(query_text, k=k*2)
        
        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60  # RRF constant
        fused_scores = {}
        
        # Score from dense search
        for rank, result in enumerate(dense_results, 1):
            rid = result['id']
            fused_scores[rid] = fused_scores.get(rid, 0) + \
                alpha / (rrf_k + rank)
        
        # Score from sparse search
        for rank, result in enumerate(sparse_results, 1):
            rid = result['id']
            fused_scores[rid] = fused_scores.get(rid, 0) + \
                (1 - alpha) / (rrf_k + rank)
        
        # Sort by fused score
        sorted_ids = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k*2]
        
        # Retrieve full documents
        candidates = []
        all_results = {r['id']: r for r in dense_results + sparse_results}
        
        for rid, score in sorted_ids:
            if rid in all_results:
                candidates.append({
                    **all_results[rid],
                    'fusion_score': score
                })
        
        # Rerank with cross-encoder
        if use_reranking and len(candidates) > 0:
            candidates = self._rerank(query_text, candidates, k)
        
        return candidates[:k]
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        k: int
    ) -> List[Dict]:
        """
        Rerank candidates using a cross-encoder for maximum precision.
        """
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['text']] for doc in candidates]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Add rerank scores and sort
        for doc, score in zip(candidates, rerank_scores):
            doc['rerank_score'] = float(score)
        
        reranked = sorted(
            candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:k]
    
    def multi_vector_search(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        k: int = 10,
        text_weight: float = 0.6
    ) -> List[Dict]:
        """
        Search using separate text and image embeddings with weighted fusion.
        Allows for fine-grained multimodal matching.
        """
        
        # Search with text embedding
        text_results = self.dense_search(text_embedding, k=k*2)
        
        # Search with image embedding
        image_results = self.dense_search(image_embedding, k=k*2)
        
        # Weighted fusion
        combined_scores = {}
        
        for result in text_results:
            rid = result['id']
            combined_scores[rid] = {
                'score': result['score'] * text_weight,
                'data': result
            }
        
        for result in image_results:
            rid = result['id']
            if rid in combined_scores:
                combined_scores[rid]['score'] += \
                    result['score'] * (1 - text_weight)
            else:
                combined_scores[rid] = {
                    'score': result['score'] * (1 - text_weight),
                    'data': result
                }
        
        # Sort and return
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return [
            {**item[1]['data'], 'combined_score': item[1]['score']}
            for item in sorted_results[:k]
        ]
    
    def diversity_reranking(
        self,
        results: List[Dict],
        k: int = 10,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR) for diverse results.
        Avoids returning too many similar reviews.
        """
        
        if len(results) == 0:
            return []
        
        selected = [results[0]]  # Start with top result
        remaining = results[1:]
        
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.get('rerank_score', candidate.get('score', 0))
                
                # Similarity to already selected (using embeddings)
                max_sim = max([
                    self._cosine_similarity(
                        candidate.get('embedding', []),
                        selected_doc.get('embedding', [])
                    )
                    for selected_doc in selected
                ])
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select best MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))