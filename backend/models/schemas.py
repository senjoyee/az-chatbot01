from datetime import datetime
from typing import Dict, List, Optional, Any
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
    chat_history: List[Message]
    documents: Optional[List[Any]] = None
    response: Optional[str] = None
    # Web search state
    waiting_for_web_search_confirmation: bool = False
    web_documents: Optional[List[Document]] = None
    web_search_query: Optional[str] = None
    customer: Optional[str] = None
    conversation_turns: int = 0
    should_stop: bool = False
    can_generate_answer: bool = True
    answer_generated_from_document_store: Optional[str] = None