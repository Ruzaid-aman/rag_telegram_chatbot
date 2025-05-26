# Chunk messages with metadata

import logging
from typing import List, Dict, Generator
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chunking.log'),
        logging.StreamHandler()
    ]
)

def extract_message_text(msg: Dict) -> str:
    """Extract text content from a message, handling different message formats"""
    text = msg.get('text', '')
    
    # Handle cases where text is a list (formatted messages)
    if isinstance(text, list):
        # Process each element and convert to string
        processed_text = []
        for item in text:
            if isinstance(item, str):
                processed_text.append(item)
            elif isinstance(item, dict) and 'text' in item:
                processed_text.append(item['text'])
        text = ' '.join(processed_text)
    
    return str(text)

def extract_participants(messages: List[Dict], index: int, window_size: int = 5) -> List[str]:
    """Extract participants from a window of messages around the current message"""
    start = max(0, index - window_size)
    end = min(len(messages), index + window_size + 1)
    window = messages[start:end]
    participants = set()
    
    for msg in window:
        # Extract participant from different possible fields
        from_user = msg.get('from', '')
        actor = msg.get('actor', '')
        if from_user:
            participants.add(from_user)
        if actor:
            participants.add(actor)
    
    return list(participants)

def extract_thread_context(messages: List[Dict], index: int, context_size: int = 3) -> str:
    """Extract context from previous messages in the thread"""
    start = max(0, index - context_size)
    context_messages = messages[start:index]
    context = []
    
    for msg in context_messages:
        text = extract_message_text(msg)
        from_user = msg.get('from', 'Unknown')
        if text:
            context.append(f"{from_user}: {text}")
    
    return '\n'.join(context)

def chunk_messages(messages: List[Dict], batch_size: int = 100) -> Generator[Dict, None, None]:
    """Process messages in batches to optimize memory usage"""
    try:
        logging.info(f"Starting to chunk {len(messages)} messages")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", ".", "!", "?", " "],
            keep_separator=True
        )
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} ({len(batch)} messages)")
            
            for j, msg in enumerate(batch):
                try:
                    # Extract message text
                    text = extract_message_text(msg)
                    if not text.strip():
                        continue
                    
                    # Create metadata
                    metadata = {
                        'message_id': msg.get('id', ''),
                        'timestamp': msg.get('date', ''),
                        'from_user': msg.get('from', ''),
                        'participants': extract_participants(messages, i + j),
                        'thread_context': extract_thread_context(messages, i + j),
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # Create context-rich text
                    text_with_context = (
                        f"Time: {metadata['timestamp']}\n"
                        f"From: {metadata['from_user']}\n"
                        f"Participants: {', '.join(metadata['participants'])}\n"
                        f"Previous context: {metadata['thread_context']}\n"
                        f"Message: {text}"
                    )
                    
                    # Split into chunks
                    chunks = splitter.create_documents([text_with_context])
                    for chunk in chunks:
                        chunk_dict = {
                            'text': chunk.page_content,
                            'metadata': {
                                **metadata,
                                'text': chunk.page_content,  # Include the text in metadata
                                'chunk_index': chunks.index(chunk)
                            }
                        }
                        yield chunk_dict
                        
                except Exception as e:
                    logging.error(f"Error processing message {i+j}: {str(e)}")
                    continue
                    
    except Exception as e:
        logging.error(f"Error in chunk_messages: {str(e)}")
        raise

