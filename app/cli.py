import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from app.retriever import retrieve_relevant_chunks
from app.llm_interface import llm
from app.vector_store import vector_store
from app.data_loader import process_new_export
import os

# Add the project directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cli.log'),
        logging.StreamHandler()
    ]
)

console = Console()

def load_vector_store():
    """Load the vector store if it exists"""
    index_path = "data/embeddings/faiss_index.idx"
    if os.path.exists(index_path):
        try:
            vector_store.load_index(index_path)
            console.print("[green]Loaded existing vector store[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error loading vector store: {str(e)}[/red]")
    return False

def main():
    console.print(Panel.fit("Telegram RAG Chatbot", style="bold magenta"))
    
    # Process data
    file_path = "data/raw_exports/VIP SUPPORT BROWN COFFEE PAYWAY INTEGRATION.json"  # Replace with your file
    if not process_new_export(file_path):
        console.print("[red]Failed to process data. Check logs.[/red]")
        return
    
    # Check if vector store exists
    if not load_vector_store():
        console.print("[red]No processed data found. Please process a Telegram export file first.[/red]")
        return
    
    console.print("\nType your questions about the chat history (or 'exit' to quit).")
    
    while True:
        try:
            query = console.input("\n[bold blue]You:[/bold blue] ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            console.print("\n[bold yellow]Searching...[/bold yellow]")
            results = retrieve_relevant_chunks(query)
            
            if not results:
                console.print("[red]No relevant information found.[/red]")
                continue
            
            # Show relevant snippets
            console.print("\n[cyan]Most relevant context:[/cyan]")
            for i, (meta, score) in enumerate(results[:2], 1):
                snippet = meta.get('text', '')[:200] + "..."
                console.print(f"\n{i}. [Score: {score:.3f}]\n{snippet}")
            
            # Generate response
            context_chunks = [{'text': meta['text']} for meta, _ in results]
            answer = llm.generate_answer(query, context_chunks)
            
            console.print("\n[bold green]Bot:[/bold green]", Panel(answer, style="green"))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            console.print(f"[red]An error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main()
