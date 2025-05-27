import requests
from typing import List, Dict
from app.logging_config import setup_logging
import json

logger = setup_logging('llm')

class OllamaLLM:    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model = "gemma3:1b"  # Using available model
        self.context_window = 8192  # Maximum context length
        self.max_tokens = 1024  # Maximum response length
        
    def _format_context(self, context_chunks: List[Dict], max_length: int = 4096) -> str:
        """Format context chunks more efficiently with length control"""
        formatted_chunks = []
        current_length = 0
        
        for chunk in context_chunks:
            text = chunk.get('text', '')
            if not text:
                continue
                
            # Format chunk with minimal metadata
            metadata = chunk.get('metadata', {})
            timestamp = metadata.get('timestamp', '')
            
            chunk_text = f"Time: {timestamp}\nContent: {text}\n\n"
            chunk_length = len(chunk_text)
            
            # Check if adding this chunk would exceed max length
            if current_length + chunk_length > max_length:
                break
                
            formatted_chunks.append(chunk_text)
            current_length += chunk_length
        
        return "".join(formatted_chunks).strip()

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer with optimized context handling and error recovery"""
        try:
            # Format context within limits
            context = self._format_context(context_chunks, max_length=self.context_window // 2)
            
            # Construct optimized prompt
            prompt = f"""Use the following chat history context to answer the question.
            Be concise and only use information from the provided context.
            If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {query}

            Answer:"""

            # Call Ollama API with optimized parameters
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "stop": ["Question:", "Context:"]
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Sorry, I encountered an error while generating the response. Please try again."
            
            answer = response.json().get('response', '').strip()
            if not answer:
                return "I couldn't generate a relevant answer from the provided context."
            
            return answer
            
        except requests.Timeout:
            logger.error("Ollama API timeout")
            return "Sorry, the response took too long. Please try a simpler question."
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, an error occurred while generating the response. Please try again."

# Create singleton instance
llm = OllamaLLM()
