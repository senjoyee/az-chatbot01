"""
Routes using native Azure AI Search SDK.
This provides parallel routes to the existing ones, allowing for gradual migration.
"""

from fastapi import APIRouter, Depends
from typing import List

from models.schemas import ConversationRequest, BlobEvent, DocumentIn
from controllers.conversation_controller_native import ConversationControllerNative
from controllers.document_controller_native import DocumentControllerNative

# Create a router with a unique prefix to avoid conflicts
router = APIRouter(prefix="/native", tags=["Native SDK Implementation"])

# Conversation routes
@router.post("/conversation")
async def ask_question(request: ConversationRequest, controller: ConversationControllerNative = Depends()):
    """
    Process a question using the native implementation.
    """
    return await controller.ask_question(request)

# Document routes
@router.post("/process_uploaded_files")
async def process_uploaded_files(event: BlobEvent, controller: DocumentControllerNative = Depends()):
    """
    Process a file that was previously uploaded using the native implementation.
    """
    return await controller.process_uploaded_files(event)

@router.post("/index_documents")
async def index_documents(documents_in: List[DocumentIn], controller: DocumentControllerNative = Depends()):
    """
    Index documents in the vector store using the native implementation.
    """
    return await controller.index_documents(documents_in)

@router.delete("/documents/{filename}")
async def delete_document(filename: str, controller: DocumentControllerNative = Depends()):
    """
    Delete a document from the vector store using the native implementation.
    """
    return await controller.delete_document(filename)
