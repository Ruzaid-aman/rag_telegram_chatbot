import logging
from app.embedder import embedder
from app.vector_store import vector_store
import numpy as np

logging.basicConfig(
    filename='logs/retriever.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    try:
        # Generate query embedding
        query_embedding = embedder.embed_query(query)
        
        # Get similar chunks
        results = vector_store.similarity_search(query_embedding, k=top_k)
        
        # Return list of (metadata, score) tuples, ensuring text is included
        processed_results = []
        for metadata, score in results:
            # Include the text in the metadata for the LLM
            if 'text' not in metadata and metadata.get('message_text'):
                metadata['text'] = metadata['message_text']
            processed_results.append((metadata, score))
            
        return processed_results
        
    except Exception as e:
        logging.error(f"Error retrieving chunks: {str(e)}")
        return []
