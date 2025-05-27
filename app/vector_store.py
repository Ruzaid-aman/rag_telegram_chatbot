# Manage FAISS or TinyVectorDB index
import numpy as np
import pickle
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from tiny_vectordb import VectorDatabase
from app.embedder import embedder
from app.logging_config import setup_logging

logger = setup_logging('vector_store')

class VectorStore:
    def __init__(self, dim: int = 384):
        """Initialize with TinyVectorDB for better memory efficiency"""
        try:
            db_path = Path("data/embeddings/vector_store.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = VectorDatabase(vector_size=dim, database_path=str(db_path))
            self.metadata = {}
            logger.info(f"Initialized VectorStore with dim={dim}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise

    def add_vectors(self, vectors: np.ndarray, chunks: List[Dict]):
        """Add vectors with optimized batch insertion"""
        try:
            for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
                vector_id = self.db.insert(vector.tolist())
                chunk['metadata'].update({
                    'vector_id': vector_id,
                    'indexed_at': datetime.now().isoformat()
                })
                self.metadata[vector_id] = chunk['metadata']
            
            logger.info(f"Added batch of {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {str(e)}")
            raise

    def similarity_search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform similarity search with TinyVectorDB's optimized algorithm"""
        try:
            results = self.db.query(query_vector.tolist(), k=k)
            return [(self.metadata.get(vid, {}), score) for vid, score in results]
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

    def save_index(self, path: str):
        """Save metadata (TinyVectorDB handles its own persistence)"""
        try:
            meta_path = Path(path).with_suffix('.meta')
            with meta_path.open('wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'vectorizer_state': {
                        'vectorizer': embedder.vectorizer,
                        'svd': embedder.svd,
                        'fitted': embedder.fitted,
                        'svd_fitted': hasattr(embedder, 'svd_fitted')
                    }
                }, f)
            logger.info(f"Saved metadata to {meta_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    def load_index(self, path: str):
        """Load metadata (TinyVectorDB handles its own persistence)"""
        try:
            meta_path = Path(path).with_suffix('.meta')
            with meta_path.open('rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                
                # Restore embedder state
                vectorizer_state = data['vectorizer_state']
                embedder.vectorizer = vectorizer_state['vectorizer']
                embedder.svd = vectorizer_state['svd']
                embedder.fitted = vectorizer_state['fitted']
                if vectorizer_state['svd_fitted']:
                    embedder.svd_fitted = True
            
            logger.info(f"Loaded metadata from {meta_path}")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise

# Create singleton instance
vector_store = VectorStore()
