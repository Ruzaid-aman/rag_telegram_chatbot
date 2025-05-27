import logging
import sys
import time
import psutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from app.retriever import retrieve_relevant_chunks
from app.llm_interface import llm
from app.vector_store import vector_store
from app.data_loader import process_new_export
from app.logging_config import setup_logging
import os

logger = setup_logging('cli')
console = Console()

class PerformanceStats:
    def __init__(self):
        self.query_times = []
        self.retrieval_times = []
        self.generation_times = []
        self.total_queries = 0
        
    def add_timing(self, query_time: float, retrieval_time: float, generation_time: float):
        self.query_times.append(query_time)
        self.retrieval_times.append(retrieval_time)
        self.generation_times.append(generation_time)
        self.total_queries += 1
        
    def get_stats_table(self) -> Table:
        if not self.query_times:
            return None
            
        table = Table(title="Performance Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Average", justify="right", style="green")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        
        metrics = {
            "Total Query Time": self.query_times,
            "Retrieval Time": self.retrieval_times,
            "Generation Time": self.generation_times
        }
        
        for name, times in metrics.items():
            if times:
                avg = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                table.add_row(
                    name,
                    f"{avg:.2f}s",
                    f"{min_time:.2f}s",
                    f"{max_time:.2f}s"
                )
                
        return table

stats = PerformanceStats()

def load_vector_store() -> bool:
    """Load vector store with progress indication"""
    index_path = "data/embeddings/vector_store.db"
    try:
        if Path(index_path).exists():
            with console.status("[bold green]Loading vector store..."):
                vector_store.load_index(index_path)
            console.print("[green]✓ Loaded existing vector store[/green]")
            return True
        return False
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        console.print(f"[red]Error loading vector store: {str(e)}[/red]")
        return False

def process_query(query: str):
    """Process a single query with timing and error handling"""
    try:
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        results = retrieve_relevant_chunks(query)
        retrieval_time = time.time() - retrieval_start
        
        if not results:
            console.print("[yellow]No relevant information found in the chat history.[/yellow]")
            return
        
        # Show context snippets
        console.print("\n[cyan]Most relevant context:[/cyan]")
        for i, (meta, score) in enumerate(results[:2], 1):
            snippet = meta.get('text', '')[:200] + "..."
            console.print(f"\n{i}. [dim][Score: {score:.3f}][/dim]\n{snippet}")
        
        # Generate response
        generation_start = time.time()
        context_chunks = [{'text': meta['text']} for meta, _ in results]
        answer = llm.generate_answer(query, context_chunks)
        generation_time = time.time() - generation_start
        
        # Show response
        console.print("\n[bold green]Bot:[/bold green]", Panel(answer, style="green"))
        
        # Update stats
        total_time = time.time() - start_time
        stats.add_timing(total_time, retrieval_time, generation_time)
        console.print(f"[dim]Processed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)[/dim]")
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        console.print(f"[red]An error occurred: {str(e)}[/red]")

def show_memory_usage():
    """Show current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    console.print(f"\n[blue]Current memory usage: {memory_mb:.1f} MB[/blue]")

def main():
    console.print(Panel.fit("Telegram RAG Chatbot", style="bold magenta"))
    
    # Process data if needed
    file_path = "data/raw_exports/VIP SUPPORT BROWN COFFEE PAYWAY INTEGRATION.json"
    if not Path("data/embeddings/vector_store.db").exists():
        if not process_new_export(file_path):
            console.print("[red]Failed to process data. Check logs.[/red]")
            return
    
    # Load vector store
    if not load_vector_store():
        console.print("[red]No processed data found. Please process a Telegram export file first.[/red]")
        return
    
    console.print("\nCommands:")
    console.print("- Type your question and press Enter to search chat history")
    console.print("- Type [bold]stats[/bold] to see performance statistics")
    console.print("- Type [bold]memory[/bold] to see memory usage")
    console.print("- Type [bold]exit[/bold] to quit")
    
    while True:
        try:
            query = Prompt.ask("\n[bold blue]You").strip().lower()
            
            if query == 'exit':
                break
            elif query == 'stats':
                if stats_table := stats.get_stats_table():
                    console.print(stats_table)
                else:
                    console.print("[yellow]No queries processed yet.[/yellow]")
                continue
            elif query == 'memory':
                show_memory_usage()
                continue
            elif not query:
                continue
            
            process_query(query)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")

if __name__ == "__main__":
    main()
