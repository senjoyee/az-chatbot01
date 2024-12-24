"""
Pydantic models for the application.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Model for chat messages."""
    role: str
    content: str

class Conversation(BaseModel):
    """Model for conversation history."""
    conversation: List[Message]

class ConversationRequest(BaseModel):
    """Model for conversation requests."""
    question: str
    conversation: Conversation

class DocumentIn(BaseModel):
    """Model for document input."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BlobEvent(BaseModel):
    """Model for blob storage events."""
    event_type: str
    file_name: str

class FileProcessingStatus(BaseModel):
    """Model for file processing status."""
    status: str  # Using str instead of ProcessingStatus to avoid circular imports
    file_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
