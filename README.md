# Telegram RAG Chatbot

This project turns Telegram chat exports into a context-aware assistant that can answer questions about the chat history. The workflow is fully local and uses FAISS for vector search and an Ollama-hosted model for response generation.

## Project Structure

- **app/** – Python modules implementing the pipeline
  - `data_loader.py` – parse Telegram JSON exports, create chunks and embeddings, and save to the vector index
  - `chunker.py` – split messages into context-aware chunks with metadata
  - `embedder.py` – create lightweight TF‑IDF embeddings
  - `vector_store.py` – FAISS-based vector store with save/load helpers
  - `retriever.py` – retrieve relevant chunks for a query
  - `llm_interface.py` – query the local LLM via Ollama (configured for `gemma3:1b`)
  - `cli.py` – command-line chat interface
- **data/** – raw exports, processed chunks, and vector indices
- **logs/** – log files for troubleshooting

`run_chatbot.sh` provides a convenient entry point for the CLI.

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your Telegram export JSON file in `data/raw_exports/`.
3. Run the chatbot:
   ```bash
   ./run_chatbot.sh
   ```
   The script processes `VIP SUPPORT BROWN COFFEE PAYWAY INTEGRATION.json` by default. Adjust `app/cli.py` if you wish to use a different file.

`test_rag.py` demonstrates an end-to-end run that processes the export, retrieves context, and queries the model.
