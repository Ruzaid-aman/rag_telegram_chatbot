from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from app.logging_config import setup_logging

logger = setup_logging('embedder')

class Embedder:
    def __init__(self, max_features: int = 5000, n_components: int = 384):
        """Initialize with optimized parameters for 16GB RAM systems"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'  # Simpler token pattern
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.fitted = False
        
    def _ensure_dimension(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensure embeddings have the correct dimension"""
        target_dim = self.svd.n_components
        current_dim = embeddings.shape[1]
        
        if current_dim < target_dim:
            padding = np.zeros((embeddings.shape[0], target_dim - current_dim))
            return np.hstack((embeddings, padding))
        elif current_dim > target_dim:
            return embeddings[:, :target_dim]
        return embeddings

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 1000) -> np.ndarray:
        """Generate embeddings in memory-efficient batches"""
        try:
            texts = [chunk['text'] for chunk in chunks]
            all_embeddings = []
            
            # Process in batches to manage memory
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate TF-IDF vectors for batch
                if not self.fitted:
                    tfidf_matrix = self.vectorizer.fit_transform(batch_texts)
                    self.fitted = True
                else:
                    tfidf_matrix = self.vectorizer.transform(batch_texts)
                
                # Reduce dimensionality for batch
                if not hasattr(self, 'svd_fitted'):
                    batch_embeddings = self.svd.fit_transform(tfidf_matrix)
                    self.svd_fitted = True
                else:
                    batch_embeddings = self.svd.transform(tfidf_matrix)
                
                # Ensure correct dimension and normalize
                batch_embeddings = self._ensure_dimension(batch_embeddings)
                batch_embeddings = normalize(batch_embeddings)
                all_embeddings.append(batch_embeddings)
                
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings for {len(chunks)} chunks with shape {embeddings.shape}")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        try:
            if not self.fitted:
                raise RuntimeError("Vectorizer not fitted. Process some chunks first.")
                
            query_vec = self.vectorizer.transform([query])
            query_embedding = self.svd.transform(query_vec)
            query_embedding = self._ensure_dimension(query_embedding)
            query_embedding = normalize(query_embedding)
            
            return query_embedding[0].astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

# Create singleton instance
embedder = Embedder()
