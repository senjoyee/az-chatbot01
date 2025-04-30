from fastapi import APIRouter, Depends, Path
from typing import List, Dict

from models.schemas import BlobEvent, DocumentIn
from controllers.document_controller import DocumentController

router = APIRouter(tags=["Document Operations"])

@router.post("/process_uploaded_files")
async def process_uploaded_files(event: BlobEvent, controller: DocumentController = Depends()):
    """
    Process a file that was previously uploaded.
    """
    return await controller.process_uploaded_files(event)

@router.post("/index_documents/")
async def index_documents(documents_in: List[DocumentIn], controller: DocumentController = Depends()):
    """
    Index documents in the vector store.
    """
    return await controller.index_documents(documents_in)

@router.get("/summarize/{filename}")
async def summarize_document(filename: str = Path(..., description="The filename to summarize"), controller: DocumentController = Depends()):
    """
    Generate a summary for the specified document.
    """
    return await controller.summarize_document(filename)

@router.get("/mindmap/{filename}")
async def generate_mindmap(filename: str = Path(..., description="The filename to create a mind map for"), controller: DocumentController = Depends()):
    """
    Generate a mind map visualization data for the specified document.
    """
    return await controller.generate_mindmap(filename)