from fastapi import APIRouter, Depends
from typing import List

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