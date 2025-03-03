from fastapi import APIRouter, File, HTTPException, UploadFile, Query, Form, Depends
from typing import List

from models.schemas import BlobEvent, DocumentIn
from controllers.file_controller import FileController

router = APIRouter(tags=["File Operations"])

@router.get("/listfiles")
async def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    controller: FileController = Depends()
):
    """
    List all files in the storage with pagination.
    """
    return await controller.list_files(page, page_size)

@router.delete("/deletefile/{filename}")
async def delete_file(filename: str, controller: FileController = Depends()):
    """
    Delete a file and its associated data from storage and vector store.
    """
    return await controller.delete_file(filename)

@router.post("/uploadfiles")
async def upload_files(
    files: List[UploadFile] = File(...),
    customer_map: str = Form(None),
    controller: FileController = Depends()
):
    """
    Upload multiple files with optional customer mapping.
    """
    return await controller.upload_files(files, customer_map)

@router.post("/process_uploaded_files")
async def process_uploaded_files(event: BlobEvent, controller: FileController = Depends()):
    """
    Process a file that was previously uploaded.
    """
    return await controller.process_uploaded_files(event)

@router.post("/index_documents/")
async def index_documents(documents_in: List[DocumentIn], controller: FileController = Depends()):
    """
    Index documents in the vector store.
    """
    return await controller.index_documents(documents_in)