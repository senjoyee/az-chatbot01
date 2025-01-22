from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from azure.data.tables import TableServiceClient
from datetime import datetime, timedelta
import os
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class ProcessingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class AzureStorageManager:
    def __init__(self):
        connection_string = os.getenv("BLOB_CONN_STRING")
        if not connection_string:
            raise ValueError("BLOB_CONN_STRING environment variable is not set")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.table_service_client = TableServiceClient.from_connection_string(connection_string)
        
        # Ensure status table exists
        self.table_client = self.table_service_client.create_table_if_not_exists("fileprocessingstatus")
        
        # Get container name from environment variable
        self.container_name = os.getenv("BLOB_CONTAINER")
        if not self.container_name:
            raise ValueError("BLOB_CONTAINER environment variable is not set")
            
        logger.info(f"Using blob container '{self.container_name}'")
    
    async def check_and_acquire_processing_lock(self, container_name: str, blob_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a file is being processed and try to acquire a lock.
        Returns (can_process, error_message)
        """
        try:
            # First check the current status
            status, error = await self.get_status(blob_name)
            if status == ProcessingStatus.IN_PROGRESS:
                return False, "File is already being processed"
            
            # Update status to in_progress
            await self.update_status(blob_name, ProcessingStatus.IN_PROGRESS)
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking processing status: {str(e)}")
            return False, str(e)
    
    async def update_status(self, file_name: str, status: ProcessingStatus, error_message: Optional[str] = None):
        """Update the processing status of a file in Azure Table Storage."""
        try:
            now = datetime.utcnow()
            entity = {
                "PartitionKey": "FileStatus",
                "RowKey": file_name,
                "Status": status.value,
                "LastUpdated": now.isoformat(),
                "ErrorMessage": error_message or ""
            }
            
            # Set start_time when moving to IN_PROGRESS
            if status == ProcessingStatus.IN_PROGRESS:
                entity["StartTime"] = now.isoformat()
                entity["EndTime"] = None
            
            # Set end_time when completing or failing
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                entity["EndTime"] = now.isoformat()
            
            # Reset times when going back to NOT_STARTED
            elif status == ProcessingStatus.NOT_STARTED:
                entity["StartTime"] = None
                entity["EndTime"] = None
            
            logger.info(f"Updating status for {file_name} to {status.value}")
            self.table_client.upsert_entity(entity)
            logger.info(f"Successfully updated status for {file_name}")
        except Exception as e:
            logger.error(f"Error updating status for {file_name}: {str(e)}")
            raise
    
    async def get_status(self, file_name: str) -> tuple[ProcessingStatus, Optional[str]]:
        """Get the current processing status of a file."""
        try:
            entity = self.table_client.get_entity("FileStatus", file_name)
            return ProcessingStatus(entity["Status"]), entity.get("ErrorMessage")
        except:
            return ProcessingStatus.NOT_STARTED, None

    async def reset_status(self, file_name: str):
        """Reset the processing status of a file to NOT_STARTED."""
        try:
            entity = {
                "PartitionKey": "FileStatus",
                "RowKey": file_name,
                "Status": ProcessingStatus.NOT_STARTED.value,
                "LastUpdated": datetime.utcnow().isoformat(),
                "ErrorMessage": ""
            }
            self.table_client.upsert_entity(entity)
            logger.info(f"Reset status for file {file_name} to NOT_STARTED")
        except Exception as e:
            logger.error(f"Error resetting status for {file_name}: {str(e)}")
            raise

    async def delete_status(self, file_name: str):
        """Delete the processing status entry for a file."""
        try:
            self.table_client.delete_entity(partition_key="FileStatus", row_key=file_name)
            logger.info(f"Deleted status entry for file {file_name}")
        except Exception as e:
            # If the entity doesn't exist, that's fine
            if "ResourceNotFound" in str(e):
                logger.info(f"No status entry found for file {file_name}")
                return
            logger.error(f"Error deleting status for {file_name}: {str(e)}")
            raise

    async def get_all_statuses(self, status_filter: Optional[ProcessingStatus] = None) -> list[dict]:
        """Get all file processing statuses with optional filtering."""
        try:
            # Build filter query if status filter is provided
            filter_query = f"PartitionKey eq 'FileStatus'"
            if status_filter:
                filter_query += f" and Status eq '{status_filter.value}'"
            
            # Query table
            entities = self.table_client.query_entities(filter_query)
            
            # Convert to list and format timestamps
            results = []
            for entity in entities:
                status_entry = {
                    "file_name": entity["RowKey"],
                    "status": entity["Status"],
                    "last_updated": entity["LastUpdated"],
                    "error_message": entity.get("ErrorMessage", ""),
                    "start_time": entity.get("StartTime"),
                    "end_time": entity.get("EndTime")
                }
                results.append(status_entry)
            
            return results
        except Exception as e:
            logger.error(f"Error getting file statuses: {str(e)}")
            raise

    async def download_file(self, file_name: str, container_name: str = None) -> str:
        """
        Download a blob to a temporary file and return the file path.
        
        Args:
            file_name: Name of the blob to download
            container_name: Name of the container (defaults to self.container_name)
            
        Returns:
            str: Path to the temporary file containing the downloaded blob
        """
        try:
            # Use default container if none specified
            container_name = container_name or self.container_name
            
            # Get container and blob clients
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(file_name)
            
            # Create a temporary file with the same extension as the blob
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                # Download the blob
                download_stream = blob_client.download_blob()
                temp_file.write(download_stream.readall())
                temp_file.flush()
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Error downloading file {file_name}: {str(e)}")
            raise
