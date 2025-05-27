"""Centralized logging configuration"""
import logging
from pathlib import Path

def setup_logging(module_name: str) -> logging.Logger:
    """Set up logging for a module with both file and console handlers"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(f'logs/{module_name}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
