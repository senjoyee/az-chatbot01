from typing import Optional, List
from fastapi import HTTPException
from datetime import datetime

from models.enums import ProcessingStatus
from models.schemas import FileProcessingStatus, BlobEvent
from services.azure_storage import AzureStorageManager
from services.file_processor import FileProcessor
from services.document_service import DocumentService

class FileController:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.file_processor = FileProcessor()
        self.document_service = DocumentService()

    async def get_file_status(self, file_name: str) -> FileProcessingStatus:
        status, error_message = await self.storage_manager.get_status(file_name)
        return FileProcessingStatus(
            status=status,
            file_name=file_name,
            error_message=error_message
        )

    async def get_file_statuses(self, page: int, page_size: int, status_filter: Optional[ProcessingStatus] = None):
        try:
            all_statuses = await self.storage_manager.get_all_statuses(status_filter)
            all_statuses.sort(key=lambda x: x["last_updated"], reverse=True)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_statuses = all_statuses[start_idx:end_idx]
            return {
                "total": len(all_statuses),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(all_statuses) + page_size - 1) // page_size,
                "statuses": paginated_statuses
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def reset_file_status(self, filename: str):
        try:
            await self.storage_manager.reset_status(filename)
            return {"message": f"Successfully reset status for {filename}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def process_uploaded_files(self, event: BlobEvent):
        file_name = event.file_name
        can_process, error_message = await self.storage_manager.check_and_acquire_processing_lock(
            self.storage_manager.container_name, file_name)
        if not can_process:
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                error_message=error_message
            )
        try:
            # Start the processing task
            import asyncio
            asyncio.create_task(self.file_processor.process_file_async(event))
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                start_time=datetime.utcnow()
            )
        except Exception as e:
            await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def list_files(self, page: int, page_size: int):
        try:
            from azure.storage.blob import BlobServiceClient
            from config.settings import BLOB_CONN_STRING, BLOB_CONTAINER
            import logging
            
            logger = logging.getLogger(__name__)
            blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
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
                status, error_message = await self.storage_manager.get_status(file_info["name"])
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
    
    async def delete_file(self, filename: str):
        try:
            from azure.storage.blob import BlobServiceClient
            from config.settings import BLOB_CONN_STRING, BLOB_CONTAINER
            import logging
            
            logger = logging.getLogger(__name__)
            logger.info(f"Attempting to delete file: {filename}")
            
            blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
            container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
            blob_client = container_client.get_blob_client(blob=filename)
            
            if not blob_client.exists():
                logger.error(f"File {filename} not found")
                raise HTTPException(status_code=404, detail=f"File {filename} not found")
                
            logger.info(f"Deleting chunks from vector store for file: {filename}")
            deletion_result = await self.document_service.delete_from_vector_store(filename)
            logger.info(f"Vector store deletion result: {deletion_result}")
            
            blob_client.delete_blob()
            logger.info(f"Successfully deleted file: {filename}")
            
            await self.storage_manager.delete_status(filename)
            logger.info(f"Successfully deleted status for file: {filename}")
            
            return {
                "message": f"File {filename} and its {deletion_result['deleted_count']} chunks deleted successfully",
                "deleted_count": deletion_result['deleted_count']
            }
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def upload_files(self, files, customer_map=None):
        import json
        import logging
        from azure.storage.blob import BlobServiceClient
        from config.settings import BLOB_CONN_STRING, BLOB_CONTAINER
        from models.enums import ProcessingStatus
        
        logger = logging.getLogger(__name__)
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
        
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
        
        # Initialize statuses for all files as NOT_STARTED
        for file in files:
            await self.storage_manager.update_status(file.filename, ProcessingStatus.NOT_STARTED)
        
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
                    await self.storage_manager.update_status(file.filename, ProcessingStatus.FAILED)
                    results.append({"filename": file.filename, "status": "error", "message": str(e)})
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                await self.storage_manager.update_status(file.filename, ProcessingStatus.FAILED)
                results.append({"filename": file.filename, "status": "error", "message": str(e)})
        
        return {"results": results}
    
    async def index_documents(self, documents_in):
        return await self.document_service.index_documents(documents_in)