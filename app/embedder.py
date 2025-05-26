import logging
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embedder.log'),
        logging.StreamHandler()
    ]
)

class Embedder:
    def __init__(self, dim: int = 384):
        """Initialize with a fixed embedding dimension"""
        try:
            self.dim = min(dim, 384)  # Ensure dimension isn't too large
            self.vectorizer = TfidfVectorizer(
                max_features=min(1000, self.dim * 2),  # Ensure enough features for SVD
                stop_words='english'
            )
            self.svd = TruncatedSVD(n_components=self.dim, random_state=42)
            self.fitted = False
            logging.info(f"Initialized TF-IDF vectorizer with dimension {self.dim}")
            
        except Exception as e:
            logging.error(f"Error initializing embedder: {str(e)}")
            raise

    def _ensure_dimension(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensure embeddings have the correct dimension"""
        current_dim = embeddings.shape[1]
        if current_dim < self.dim:
            # Pad with zeros if necessary
            padding = np.zeros((embeddings.shape[0], self.dim - current_dim))
            embeddings = np.hstack((embeddings, padding))
        elif current_dim > self.dim:
            # Truncate if necessary
            embeddings = embeddings[:, :self.dim]
        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate TF-IDF vectors
            if not self.fitted:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.fitted = True
            else:
                tfidf_matrix = self.vectorizer.transform(texts)
                
            # Reduce dimensionality
            if not hasattr(self, 'svd_fitted'):
                embeddings = self.svd.fit_transform(tfidf_matrix)
                self.svd_fitted = True
            else:
                embeddings = self.svd.transform(tfidf_matrix)
            
            # Ensure correct dimension and normalize
            embeddings = self._ensure_dimension(embeddings)
            embeddings = normalize(embeddings)
            
            logging.info(f"Generated embeddings for {len(chunks)} chunks with shape {embeddings.shape}")
            return embeddings.astype(np.float32)  # Ensure float32 for FAISS
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        try:
            if not self.fitted:
                raise RuntimeError("Vectorizer not fitted. Process some chunks first.")
                
            # Generate TF-IDF vector for query
            query_vec = self.vectorizer.transform([query])
            
            # Reduce dimensionality
            query_embedding = self.svd.transform(query_vec)
            
            # Ensure correct dimension and normalize
            query_embedding = self._ensure_dimension(query_embedding)
            query_embedding = normalize(query_embedding)
            
            return query_embedding[0].astype(np.float32)  # Ensure float32 for FAISS
            
        except Exception as e:
            logging.error(f"Error generating query embedding: {str(e)}")
            raise

# Create a singleton instance
embedder = Embedder()
