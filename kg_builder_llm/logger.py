"""Logging configuration."""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file. If None, creates logs/kg_builder_YYYYMMDD_HHMMSS.log
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"kg_builder_{timestamp}.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging to file: %s", log_file)

    # Silence verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j_graphrag").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
