"""
Logging configuration for the application.
Maintains the exact same logging setup as the original implementation.
"""

import logging
import sys

def setup_logging():
    """Configure logging with the same settings as the original implementation."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Log to stdout for Azure App Service
        ]
    )

    # Define logger configurations (exactly as in original implementation)
    loggers_config = {
        # PDF processing
        'pdfminer': logging.INFO,
        'pdfminer.pdfinterp': logging.INFO,
        'pdfminer.converter': logging.INFO,
        'pdfminer.layout': logging.INFO,
        'pdfminer.image': logging.INFO,
        
        # Document processing
        'unstructured': logging.INFO,
        'detectron2': logging.INFO,
        'PIL': logging.INFO,
        
        # OCR related
        'tesseract': logging.INFO,
        
        # Azure SDK
        'azure.core.pipeline.policies.http_logging_policy': logging.INFO,
        'azure.identity': logging.INFO
    }

    # Apply logger configurations (exactly as in original implementation)
    for logger_name, level in loggers_config.items():
        logging.getLogger(logger_name).setLevel(level)
        # Ensure each logger also logs to stdout
        logger_instance = logging.getLogger(logger_name)
        if not logger_instance.handlers:
            logger_instance.addHandler(logging.StreamHandler(sys.stdout))

    # Return the root logger for the application
    return logging.getLogger(__name__)
