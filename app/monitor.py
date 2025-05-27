# Folder monitoring automation script
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from app.data_loader import process_new_export
from app.logging_config import setup_logging

logger = setup_logging('monitor')

class NewExportHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing = False
        
    def on_created(self, event: FileCreatedEvent):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
            
        try:
            file_path = Path(event.src_path)
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.warning(f"File {file_path} is invalid or empty")
                return
                
            # Wait briefly to ensure file is fully written
            time.sleep(1)
                
            logger.info(f"Processing new export: {file_path}")
            if process_new_export(str(file_path)):
                logger.info(f"Successfully processed {file_path}")
            else:
                logger.error(f"Failed to process {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {event.src_path}: {str(e)}")

def monitor_exports_folder(path: str = "data/raw_exports/"):
    try:
        export_path = Path(path)
        export_path.mkdir(parents=True, exist_ok=True)
            
        event_handler = NewExportHandler()
        observer = Observer()
        observer.schedule(event_handler, str(export_path), recursive=False)
        observer.start()
        
        logger.info(f"Started monitoring {export_path} for new exports...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping file monitor...")
            observer.stop()
            
        observer.join()
        
    except Exception as e:
        logger.error(f"Error in monitor_exports_folder: {str(e)}")
        raise

if __name__ == "__main__":
    monitor_exports_folder()

