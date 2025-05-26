# Manage FAISS or TinyVectorDB index
import faiss
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from app.embedder import embedder

logging.basicConfig(
    filename='logs/vector_store.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VectorStore:
    def __init__(self, dim: int = 384):
        """
        Initialize the vector store with configurable parameters
        Args:
            dim: Dimension of the vectors
        """
        try:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = {}
            self.next_id = 0
            logging.info(f"Initialized VectorStore with dim={dim}, index_type=L2")
        except Exception as e:
            logging.error(f"Failed to initialize VectorStore: {str(e)}")
            raise

    def add_vectors(self, vectors: np.ndarray, chunks: List[Dict]):
        """
        Add vectors to the index in batches
        """
        try:
            # Add vectors to the index
            self.index.add(vectors)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                chunk['metadata'].update({
                    'vector_id': self.next_id + i,
                    'indexed_at': datetime.now().isoformat()
                })
                self.metadata[self.next_id + i] = chunk['metadata']
            
            logging.info(f"Added batch of {len(vectors)} vectors. Total vectors: {self.next_id + len(vectors)}")
            self.next_id += len(vectors)
            
        except Exception as e:
            logging.error(f"Error adding vectors to index: {str(e)}")
            raise

    def similarity_search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Perform similarity search with configurable parameters
        """
        try:
            D, I = self.index.search(np.array([query_vector]), k)
            results = []
            
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                results.append((self.metadata.get(idx, {}), float(dist)))
            
            return results
            
        except Exception as e:
            logging.error(f"Error performing similarity search: {str(e)}")
            raise

    def save_index(self, path: str):
        """
        Save the index and metadata to disk
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, path)
            
            # Save metadata and embedder state
            with open(f"{path}.meta", "wb") as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'next_id': self.next_id,
                    'vectorizer_state': {
                        'vectorizer': embedder.vectorizer,
                        'svd': embedder.svd,
                        'fitted': embedder.fitted,
                        'svd_fitted': hasattr(embedder, 'svd_fitted')
                    }
                }, f)
            logging.info(f"Saved index to {path} with {self.next_id} vectors")
        except Exception as e:
            logging.error(f"Error saving index to {path}: {str(e)}")
            raise

    def load_index(self, path: str):
        """
        Load the index and metadata from disk
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(path)
            
            # Load metadata and embedder state
            with open(f"{path}.meta", "rb") as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.next_id = data['next_id']
                
                # Restore embedder state
                vectorizer_state = data['vectorizer_state']
                embedder.vectorizer = vectorizer_state['vectorizer']
                embedder.svd = vectorizer_state['svd']
                embedder.fitted = vectorizer_state['fitted']
                if vectorizer_state['svd_fitted']:
                    embedder.svd_fitted = True
                    
            logging.info(f"Loaded index from {path} with {self.next_id} vectors")
        except Exception as e:
            logging.error(f"Error loading index from {path}: {str(e)}")
            raise

# Create a singleton instance
vector_store = VectorStore()
