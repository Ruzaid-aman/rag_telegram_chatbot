"""
Test script to validate the RAG system end-to-end
"""
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from app.data_loader import process_new_export
from app.retriever import retrieve_relevant_chunks
from app.llm_interface import llm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test.log'),
        logging.StreamHandler()
    ]
)

console = Console()

def validate_json_file(file_path: str) -> bool:
    """Validate that the JSON file exists and is properly formatted"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict) or 'messages' not in data:
                console.print("[red]Error: Invalid Telegram export format[/red]")
                return False
            return True
    except FileNotFoundError:
        console.print(f"[red]Error: File {file_path} not found[/red]")
        return False
    except json.JSONDecodeError:
        console.print(f"[red]Error: {file_path} is not a valid JSON file[/red]")
        return False

def test_rag_pipeline():
    # Create necessary directories
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    Path("data/processed_chunks").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # 1. Validate and process the Telegram export
    export_file = "data/raw_exports/Viwat  Default Payway integration.json"
    
    console.print(Panel.fit("Step 1: Validating and Processing Export File", 
                          style="bold blue"))
    
    if not validate_json_file(export_file):
        return
    
    console.print(f"Processing export file: {export_file}")
    success = process_new_export(export_file)
    
    if not success:
        console.print("[red]Failed to process export file[/red]")
        return
    
    console.print("[green]Successfully processed export file[/green]")
    
    # 2. Test retrieval and response generation
    test_queries = [
        "What is the main issue discussed about Payway integration?",
        "What solutions or steps were proposed for the integration?",
        "What are the key technical requirements mentioned?",
        "Were there any specific problems or errors discussed?"
    ]
    
    console.print(Panel.fit("Step 2: Testing Retrieval and Response Generation", 
                          style="bold blue"))
    
    for query in track(test_queries, description="Processing queries"):
        console.print(f"\n[yellow]Query:[/yellow] {query}")
        
        # Get relevant chunks
        results = retrieve_relevant_chunks(query)
        if not results:
            console.print("[red]No relevant chunks found[/red]")
            continue
        
        # Print snippets from retrieved chunks
        console.print("\n[cyan]Top 2 relevant context snippets:[/cyan]")
        for i, (meta, score) in enumerate(results[:2], 1):
            snippet = meta.get('text', '')[:200] + "..."
            console.print(f"{i}. [Score: {score:.3f}]\n{snippet}\n")
        
        # Generate response
        try:
            context_chunks = [{'text': meta['text']} for meta, _ in results]
            answer = llm.generate_answer(query, context_chunks)
            console.print(Panel(answer, title="Generated Answer", 
                              style="green"))
        except Exception as e:
            console.print(f"[red]Error generating response: {str(e)}[/red]")

if __name__ == "__main__":
    console.print(Panel.fit("RAG System End-to-End Test", 
                          style="bold magenta"))
    test_rag_pipeline()
