import logging
from typing import List, Dict
from datetime import datetime
import requests
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/llm.log'),
        logging.StreamHandler()
    ]
)

class OllamaLLM:
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model = "gemma3:1b"  # Exact model name as shown by 'ollama list'

    def _format_context(self, context_chunks: List[Dict]) -> str:
        formatted_chunks = []
        
        for chunk in context_chunks:
            text = chunk.get('text', '')
            if not text and isinstance(chunk.get('metadata'), dict):
                text = chunk['metadata'].get('text', '')
            
            metadata = chunk.get('metadata', {})
            timestamp = metadata.get('timestamp', '')
            participants = metadata.get('participants', [])
            
            if timestamp or participants:
                formatted_text = f"Time: {timestamp}\n"
                if participants:
                    formatted_text += f"Participants: {', '.join(participants)}\n"
                formatted_text += f"Content: {text}\n"
            else:
                formatted_text = text
            
            formatted_chunks.append(formatted_text)
        
        return "\n\n".join(formatted_chunks)

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        try:
            # Format context into readable text
            context_text = self._format_context(context_chunks)

            # Create a clear prompt for the model
            system_prompt = """You are a helpful assistant that answers questions based on provided chat history context. 
Only use information from the given context. If the context doesn't contain relevant information, say so.
Be concise and direct in your responses."""

            user_prompt = f"""Context:
{context_text}

Question: {query}"""

            # Prepare the API call to Ollama
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            }

            logging.info(f"Sending request to Ollama API at {url}")
            response = requests.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            answer = result.get("message", {}).get("content", "")
            
            if not answer:
                return "I couldn't generate an answer based on the available context."

            logging.info("Successfully generated response from Ollama")
            return answer

        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama. Please make sure Ollama is running."
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            logging.error(f"Error generating answer with Ollama: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."

# Create singleton instance
llm = OllamaLLM()
