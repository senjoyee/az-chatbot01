from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

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
    metadata: Dict = Field(default_factory=dict)

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

class AgentState(BaseModel):
    question: str
    original_question: Optional[str] = None
    chat_history: List[Message]
    documents: Optional[List[Document]] = None
    response: Optional[str] = None
    should_stop: bool = False
    conversation_turns: int = 0  # New field to track conversation progression
    customer: Optional[str] = Field(None, description="Active customer context")
    can_generate_answer: Optional[bool] = None # ADD THIS LINE