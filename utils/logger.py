"""
Logging Utility
Setup logging for the system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "FruitSorter",
    log_dir: str = "./logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with date
        log_filename = log_path / f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


class SystemLogger:
    """
    System logger with context-specific methods
    """
    
    def __init__(self, name: str = "FruitSorter"):
        """Initialize system logger"""
        self.logger = setup_logger(name)
        self.stats = {
            'detections': 0,
            'fresh_sorted': 0,
            'spoiled_sorted': 0,
            'errors': 0
        }
    
    def detection(self, fruit_type: str, confidence: float):
        """Log fruit detection"""
        self.logger.info(f"🎯 Detected: {fruit_type} (confidence: {confidence:.2%})")
        self.stats['detections'] += 1
    
    def classification(self, result: str, confidence: float, is_fresh: bool):
        """Log classification result"""
        emoji = "🍎" if is_fresh else "🍂"
        self.logger.info(f"{emoji} Classified: {result} (confidence: {confidence:.2%})")
        
        if is_fresh:
            self.stats['fresh_sorted'] += 1
        else:
            self.stats['spoiled_sorted'] += 1
    
    def sorting(self, is_fresh: bool):
        """Log sorting action"""
        direction = "LEFT (Fresh)" if is_fresh else "RIGHT (Spoiled)"
        self.logger.info(f"↔️ Sorting: {direction}")
    
    def error(self, message: str, exception: Exception = None):
        """Log error"""
        if exception:
            self.logger.error(f"❌ {message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(f"❌ {message}")
        self.stats['errors'] += 1
    
    def warning(self, message: str):
        """Log warning"""
        self.logger.warning(f"⚠️ {message}")
    
    def system_event(self, message: str):
        """Log system event"""
        self.logger.info(f"⚙️ {message}")
    
    def performance(self, fps: float, processing_time: float):
        """Log performance metrics"""
        self.logger.debug(f"📊 FPS: {fps:.1f}, Processing: {processing_time*1000:.1f}ms")
    
    def print_statistics(self):
        """Print current statistics"""
        total = self.stats['fresh_sorted'] + self.stats['spoiled_sorted']
        
        print("\n" + "=" * 60)
        print("System Statistics")
        print("=" * 60)
        print(f"Total detections: {self.stats['detections']}")
        print(f"Fresh sorted: {self.stats['fresh_sorted']} ({self.stats['fresh_sorted']/total*100 if total > 0 else 0:.1f}%)")
        print(f"Spoiled sorted: {self.stats['spoiled_sorted']} ({self.stats['spoiled_sorted']/total*100 if total > 0 else 0:.1f}%)")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 60 + "\n")


# Test code
if __name__ == '__main__':
    print("=== Logger Test ===\n")
    
    # Test basic logger
    logger = setup_logger(name="TestLogger", log_dir="./test_logs")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test system logger
    sys_logger = SystemLogger(name="TestSystem")
    
    sys_logger.system_event("System initialized")
    sys_logger.detection("apple", 0.95)
    sys_logger.classification("Fresh", 0.87, True)
    sys_logger.sorting(True)
    sys_logger.performance(8.5, 0.12)
    sys_logger.print_statistics()
    
    print("\n✅ Logger test complete!")
    print(f"   Check logs in: ./test_logs/")
