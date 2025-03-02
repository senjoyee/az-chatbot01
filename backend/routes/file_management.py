import json
import logging
from typing import List
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
import asyncio

from azure.storage.blob import BlobServiceClient
from config.settings import BLOB_CONN_STRING, BLOB_CONTAINER
from models.schemas import BlobEvent, FileProcessingStatus
from models.enums import ProcessingStatus
from azure_storage import AzureStorageManager
from services.file_processor import FileProcessor
from services.contextualizer import Contextualizer
from services.vector_store_ops import delete_from_vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
storage_manager = AzureStorageManager()
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
contextualizer = Contextualizer()
file_processor = FileProcessor(storage_manager, contextualizer)

@router.post("/uploadfiles")
async def upload_files(
    files: List[UploadFile] = File(...),
    customer_map: str = Form(None)
):
    results = []
    
    # Parse the customer map if provided
    customer_mapping = {}
    if customer_map:
        try:
            customer_mapping = json.loads(customer_map)
            logger.info(f"Received customer mapping: {customer_mapping}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing customer map: {str(e)}")
    
    # Debug logging
    logger.info(f"Received {len(files)} files")
    for i, file in enumerate(files):
        logger.info(f"File {i}: {file.filename}")
        customer = customer_mapping.get(file.filename, "unknown")
        logger.info(f"Customer for {file.filename}: {customer}")
    
    # Initialize statuses for all files as NOT_STARTED
    for file in files:
        await storage_manager.update_status(file.filename, ProcessingStatus.NOT_STARTED)
    
    # Process files sequentially
    for i, file in enumerate(files):
        try:
            # Get the customer name for this file from the mapping
            customer_name = customer_mapping.get(file.filename, "unknown")
            logger.info(f"Assigning customer '{customer_name}' to file '{file.filename}'")
            
            # Process the file with its customer name
            MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/(1024*1024)}MB"
                )
            
            try:
                logger.info(f"Uploading file {file.filename} with customer metadata: {customer_name}")
                container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
                blob_client = container_client.get_blob_client(blob=file.filename)
                blob_client.upload_blob(contents, overwrite=True, metadata={"customer": customer_name})
                
                # Verify metadata was set correctly
                blob_properties = blob_client.get_blob_properties()
                logger.info(f"Verified metadata for {file.filename}: customer={blob_properties.metadata.get('customer', 'not set')}")
                
                # Leave the status as NOT_STARTED so it can be processed asynchronously later
                results.append({"filename": file.filename, "status": "success"})
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                await storage_manager.update_status(file.filename, ProcessingStatus.FAILED)
                results.append({"filename": file.filename, "status": "error", "message": str(e)})
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            await storage_manager.update_status(file.filename, ProcessingStatus.FAILED)
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
    
    return {"results": results}

@router.post("/process_uploaded_files")
async def process_uploaded_files(event: BlobEvent):
    file_name = event.file_name
    can_process, error_message = await storage_manager.check_and_acquire_processing_lock(BLOB_CONTAINER, file_name)
    if not can_process:
        return FileProcessingStatus(
            status=ProcessingStatus.IN_PROGRESS,
            file_name=file_name,
            error_message=error_message
        )
    try:
        import asyncio
        asyncio.create_task(file_processor.process_file_async(event))
        return FileProcessingStatus(
            status=ProcessingStatus.IN_PROGRESS,
            file_name=file_name,
            start_time=datetime.utcnow()
        )
    except Exception as e:
        await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/listfiles")
async def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        blob_list = list(container_client.list_blobs())
        
        # Get all files with their basic information
        files_info = [{
            "name": blob.name,
            "size": blob.size,
            "lastModified": blob.last_modified.isoformat(),
            "contentType": blob.content_settings.content_type
        } for blob in blob_list]
        
        # Get status for all files
        for file_info in files_info:
            status, error_message = await storage_manager.get_status(file_info["name"])
            logger.info(f"File {file_info['name']} status: {status} (type: {type(status)})")
            file_info["status"] = status
            if error_message:
                file_info["errorMessage"] = error_message
        
        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            "total_files": len(files_info),
            "files": files_info[start:end],
            "page": page,
            "total_pages": (len(files_info) - 1) // page_size + 1,
        }
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/deletefile/{filename}")
async def delete_file(filename: str):
    logger.info(f"Attempting to delete file: {filename}")
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
    blob_client = container_client.get_blob_client(blob=filename)
    try:
        if not blob_client.exists():
            logger.error(f"File {filename} not found")
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        logger.info(f"Deleting chunks from vector store for file: {filename}")
        deletion_result = await delete_from_vector_store(filename)
        logger.info(f"Vector store deletion result: {deletion_result}")
        blob_client.delete_blob()
        logger.info(f"Successfully deleted file: {filename}")
        await storage_manager.delete_status(filename)
        logger.info(f"Successfully deleted status for file: {filename}")
        return {
            "message": f"File {filename} and its {deletion_result['deleted_count']} chunks deleted successfully",
            "deleted_count": deletion_result['deleted_count']
        }
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))