"""
Multiprocessing-compatible logging module
"""
import logging
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
import sys

def setup_logging():
    """Setup logging with queue for multiprocessing"""
    # Create a queue for logging
    log_queue = mp.Queue()
    
    # Setup the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create file handler (optional)
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setFormatter(formatter)
    
    # Create queue listener
    queue_listener = QueueListener(
        log_queue, 
        console_handler, 
        # file_handler,
        respect_handler_level=True
    )
    queue_listener.start()
    
    return log_queue, queue_listener

# Global logging setup
_log_queue, _listener = setup_logging()

def get_logger(name=None):
    """Get a logger that works with multiprocessing"""
    logger = logging.getLogger(name)
    
    # Only add handler if it doesn't have one already
    if not logger.handlers:
        queue_handler = QueueHandler(_log_queue)
        logger.addHandler(queue_handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Cleanup function
def cleanup_logging():
    """Stop the queue listener"""
    global _listener
    if _listener:
        _listener.stop()

# Register cleanup on exit
import atexit
atexit.register(cleanup_logging)
