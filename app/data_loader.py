# Load and parse Telegram JSON exports
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from app.chunker import chunk_messages
from app.embedder import embedder
from app.vector_store import vector_store

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loader.log'),
        logging.StreamHandler()
    ]
)

def load_telegram_json(file_path: str) -> List[Dict]:
    """
    Load and validate Telegram JSON export
    """
    try:
        logging.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON format: expected dict, got {type(data)}")
            
        if 'messages' not in data:
            raise ValueError("Invalid Telegram export format: 'messages' key not found")
            
        if not isinstance(data['messages'], list):
            raise ValueError(f"Invalid messages format: expected list, got {type(data['messages'])}")
        
        logging.info(f"Successfully loaded {len(data['messages'])} messages from {file_path}")
        return data['messages']
        
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise

def process_new_export(file_path: str, batch_size: int = 100) -> bool:
    """
    Process a new Telegram export file
    """
    try:
        logging.info(f"Starting to process file: {file_path}")
        
        # Ensure directories exist
        for dir_path in ["data/processed_chunks", "data/embeddings"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Load and validate messages
        messages = load_telegram_json(file_path)
        if not messages:
            logging.warning("No messages found in export file")
            return False
        
        logging.info("Generating chunks from messages")
        chunks = list(chunk_messages(messages))
        if not chunks:
            logging.warning("No chunks generated from messages")
            return False
            
        logging.info(f"Generated {len(chunks)} chunks")
        
        # Generate embeddings
        logging.info("Generating embeddings")
        embeddings = embedder.embed_chunks(chunks)
        logging.info(f"Generated embeddings with shape {embeddings.shape}")
        
        # Add to vector store
        logging.info("Adding vectors to store")
        vector_store.add_vectors(embeddings, chunks)
        
        # Save the vector store
        index_path = "data/embeddings/faiss_index.idx"
        logging.info(f"Saving vector store to {index_path}")
        vector_store.save_index(index_path)
        
        # Save processing info
        save_processing_info(file_path)
        
        logging.info(f"Successfully processed file {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def save_processing_info(file_path: str) -> None:
    """
    Save information about processed files
    """
    try:
        info_file = Path("data/processed_chunks/processed_files.json")
        
        processed_files = []
        if info_file.exists():
            with open(info_file, 'r') as f:
                processed_files = json.load(f)
        
        file_info = {
            'file_path': file_path,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
        
        if file_info not in processed_files:
            processed_files.append(file_info)
            
            with open(info_file, 'w') as f:
                json.dump(processed_files, f, indent=2)
            logging.info(f"Updated processing info for {file_path}")
                
    except Exception as e:
        logging.error(f"Error saving processing info for {file_path}: {str(e)}")
        # Don't raise here as this is not critical for the main functionality

