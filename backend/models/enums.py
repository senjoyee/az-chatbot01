"""
Enum definitions for the application.
"""

from enum import Enum

class ProcessingStatus(str, Enum):
    """Enum for file processing status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
