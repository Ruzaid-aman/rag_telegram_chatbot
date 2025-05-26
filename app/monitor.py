# Folder monitoring automation script
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from app.data_loader import process_new_export

logging.basicConfig(
    filename='logs/monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NewExportHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing = False
        self.queue = set()

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
            
        if not event.src_path.endswith('.json'):
            return
            
        try:
            file_path = Path(event.src_path)
            if not file_path.exists():
                logging.warning(f"File {file_path} was created but no longer exists")
                return
                
            # Wait a bit to ensure the file is fully written
            time.sleep(1)
            
            if file_path.stat().st_size == 0:
                logging.warning(f"File {file_path} is empty")
                return
                
            logging.info(f"New export detected: {file_path}")
            
            if self.processing:
                self.queue.add(str(file_path))
                logging.info(f"Added {file_path} to processing queue")
                return
                
            self.process_file(str(file_path))
            
        except Exception as e:
            logging.error(f"Error handling file {event.src_path}: {str(e)}")

    def process_file(self, file_path: str):
        try:
            self.processing = True
            if process_new_export(file_path):
                logging.info(f"Successfully processed {file_path}")
            else:
                logging.error(f"Failed to process {file_path}")
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            
        finally:
            self.processing = False
            self.process_queue()

    def process_queue(self):
        while self.queue:
            next_file = self.queue.pop()
            self.process_file(next_file)

def monitor_exports_folder(path: str = "data/raw_exports/"):
    try:
        export_path = Path(path)
        if not export_path.exists():
            export_path.mkdir(parents=True)
            logging.info(f"Created directory {export_path}")
            
        event_handler = NewExportHandler()
        observer = Observer()
        observer.schedule(event_handler, str(export_path), recursive=False)
        observer.start()
        
        logging.info(f"Started monitoring {export_path} for new exports...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping file monitor...")
            observer.stop()
            
        observer.join()
        
    except Exception as e:
        logging.error(f"Error in monitor_exports_folder: {str(e)}")
        raise

if __name__ == "__main__":
    monitor_exports_folder()

