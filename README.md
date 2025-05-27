# High-Performance Contextual RAG Telegram Chatbot

A chatbot that uses Retrieval Augmented Generation (RAG) to answer questions based on Telegram chat history. It's optimized for performance and uses local LLMs.

## Project Structure

The project is organized into the following main directories:

- **`app/`**: Core Python scripts and modules. This is where the main logic of the chatbot resides, including data loading, chunking, embedding, retrieval, and interfacing with the LLM.
- **`data/`**: Stores Telegram exports and processed data. This includes:
    - `raw_exports/`: For placing exported Telegram JSON files.
    - `processed_chunks/`: For storing chunked and metadata-enriched text files.
    - `embeddings/`: For storing vector embeddings and index files (e.g., FAISS).
- **`models/`**: Contains downloaded LLM and embedding models. These are the pre-trained models used for generating responses and creating embeddings.
- **`logs/`**: For storing logs for monitoring and errors. This helps in debugging and tracking the chatbot's performance.

## Key Python Scripts (`app/`)

This section details the roles of the main Python scripts found in the `app/` directory:

- **`data_loader.py`**: Load & orchestrate processing. This script is responsible for loading the initial Telegram JSON export, then coordinates the process of chunking the messages, generating embeddings for those chunks, and finally indexing them into the vector store.
- **`chunker.py`**: Chunk messages with metadata. This script handles the logic for breaking down long messages into smaller, contextually relevant chunks. It also extracts and attaches important metadata to each chunk.
- **`embedder.py`**: Generate vector embeddings. This script manages the setup of the chosen sentence embedding model and provides functions for encoding text chunks in batches to produce their vector representations.
- **`vector_store.py`**: Vector DB management. This script is responsible for managing the vector database (e.g., FAISS or TinyVectorDB). It includes functionalities to add new vector embeddings to the index, search the index for similar vectors, and save/load the index to/from disk.
- **`retriever.py`**: Retrieve relevant chunks. This script takes a user query, embeds it using the same model as the message chunks, and then searches the vector store to find the most relevant message chunks based on semantic similarity.
- **`llm_interface.py`**: Generate answers with local LLM. This script handles the interaction with the local Large Language Model (LLM). It's responsible for loading the LLM, preparing the input prompt (which includes the user query and the retrieved context chunks), and generating a coherent answer.
- **`cli.py`**: User interaction CLI. This script provides a command-line interface for users to interact with the chatbot. It manages the input loop, calls the retriever to get relevant context, then calls the LLM interface to generate an answer, and finally prints the answer to the console.
- **`monitor.py`**: Automate new export processing. This script provides functionality to monitor a specific folder for new Telegram data exports. When a new export file is detected, it automatically triggers the data loading and processing pipeline to incorporate the new data into the chatbot's knowledge base.

## Features

- Optimized for 16GB RAM / 100GB storage.
- Uses quantized models for efficiency.
- Employs aggressive caching.
- Utilizes specialized vector databases (e.g., TinyVectorDB).
- Aims for sub-second response times where possible.
- Focuses on high contextual accuracy in RAG pipeline.
- Implements conversation-aware chunking.
- Features a multi-level retrieval strategy.
- Uses a memory-efficient local LLM setup (e.g., Phi-3 Mini).
- Includes performance monitoring capabilities.
- Provides a streamlined Command Line Interface (CLI).

## Setup and Installation

Follow these steps to set up and run the chatbot:

1.  **Python Version:**
    *   This project is developed using Python 3. Ensure you have Python 3.x installed.

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory_name>
    ```

3.  **Install Dependencies:**
    *   Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the Language Model:**
    *   This chatbot requires a GGUF-compatible language model.
    *   The recommended model (as used in development) is `phi-3-mini-4k-instruct-q4_k_m.gguf`.
    *   Download the `phi-3-mini-4k-instruct-q4_k_m.gguf` model (or a similar GGUF model) and place it in the `models/` directory. You can find GGUF models on Hugging Face or other model repositories.
    *   If you use a different model or place it in a different location, you may need to adjust the model path in `app/llm_interface.py`.

## Running the Chatbot

Follow these instructions to run the chatbot after completing the setup:

1.  **Place Telegram Data:**
    *   Export your Telegram chat history as a JSON file.
    *   Place the exported JSON file(s) into the `data/raw_exports/` directory.
    *   The `app/cli.py` script is currently configured to look for an example file named `VIP SUPPORT BROWN COFFEE PAYWAY INTEGRATION.json` in this directory if no existing processed data is found.

2.  **Initial Data Processing:**
    *   **Automatic Processing:** If you are running the chatbot for the first time, or if the processed vector store (`data/embeddings/vector_store.db`) does not exist, the Command Line Interface (CLI) will attempt to automatically process the default Telegram export file: `data/raw_exports/VIP SUPPORT BROWN COFFEE PAYWAY INTEGRATION.json`.
    *   **Monitoring for New Files:** The `monitor.py` script (if run separately) can be used to watch the `data/raw_exports/` directory for new files and process them automatically. (Further details on `monitor.py` usage would be added if it's part of the standard workflow).
    *   **Manual Processing (Alternative):** While `app/cli.py` handles the default case, `app/data_loader.py` can be used to process specific files. For example: `python -m app.data_loader --file data/raw_exports/your_other_export.json`. *(Note: This assumes `data_loader.py` has a CLI entry point; adjust if it's only used as a module).*

3.  **Start the Chatbot:**
    *   Open your terminal, navigate to the project's root directory, and run the chatbot using:
        ```bash
        python -m app.cli
        ```
    *   Alternatively, if the `run_chatbot.sh` script is configured for your environment, you can use it:
        ```bash
        ./run_chatbot.sh
        ```

4.  **CLI Commands:**
    Once the chatbot is running, you can use the following commands in the CLI:
    *   **Type your question and press Enter:** To ask the chatbot a question about your Telegram history.
    *   **`stats`**: To display performance statistics (e.g., query time, retrieval time).
    *   **`memory`**: To show the current memory usage of the chatbot process.
    *   **`exit`**: To quit the chatbot.

## Performance Benchmarks

The following are expected performance figures on a 16GB RAM configuration, as outlined in the project's development notes (`projectnotes.md`).

| Component                  | Expected Performance   | Memory Usage   |
|----------------------------|------------------------|----------------|
| Embedding Model (Quantized)| 50ms per query         | ~800MB         |
| Vector Search (TinyVectorDB)| 5ms for 100k vectors   | ~2GB           |
| LLM Generation (Phi-3 Mini)| 15 tokens/sec          | ~2.4GB         |
| **Total System (approx.)** | **2-3s end-to-end**    | **~8GB peak**  |

*Note: Actual performance may vary based on the specific hardware, data size, and query complexity.*

## Performance Monitoring

The chatbot's Command Line Interface (CLI) provides a simple way to monitor its performance during a session.

To view performance metrics:
- Typing `stats` in the CLI will display a table with performance statistics for the current session.
- This includes metrics like average, minimum, and maximum times for:
    - Total Query Time
    - Retrieval Time
    - Generation Time

These statistics can help in understanding the chatbot's responsiveness and identifying potential bottlenecks in the RAG pipeline (retrieval or generation stages).
