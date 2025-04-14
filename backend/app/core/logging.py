import logging
import sys
from pathlib import Path
from loguru import logger
from .config import settings


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and redirects them to Loguru.
    This allows seamless integration with libraries that use standard logging.
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where this record was issued
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """
    Configure logging with Loguru for enhanced log management.
    - Intercepts standard logging
    - Sets up console output
    - Configures file logging with rotation
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Remove default loguru handler
    logger.remove()
    
    # Set log level from environment variable
    log_level = settings.LOG_LEVEL.upper()

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Add file handler with rotation
    logger.add(
        "logs/app.log",
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="1 month",  # Keep logs for 1 month
        compression="zip",  # Compress rotated logs
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Replace standard library root handlers
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    
    # Apply to standard library modules
    for logger_name in [
        "uvicorn", 
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "pymongo",
        "httpx",
        "langchain",
    ]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]

    logger.info("Logging system initialized")
    return logger