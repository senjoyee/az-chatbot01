import os
import logging
import asyncio
from datetime import datetime
from typing import List, Optional

from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title

from config.settings import BLOB_CONTAINER
from models.schemas import BlobEvent, DocumentIn
from models.enums import ProcessingStatus
from utils.helpers import sanitize_id, extract_source, process_excel_elements
from services.contextualizer import Contextualizer
from azure_storage import AzureStorageManager

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self, storage_manager: AzureStorageManager, contextualizer: Contextualizer):
        self.storage_manager = storage_manager
        self.contextualizer = contextualizer
        self.processing_lock = asyncio.Lock()
    
    async def process_file_async(self, event: BlobEvent):
        """Process a file asynchronously after it has been uploaded to blob storage."""
        async with self.processing_lock:
            file_name = event.file_name
            logger.info(f"Starting to process file: {file_name}")
            try:
                if event.event_type == "Microsoft.Storage.BlobDeleted":
                    logger.info(f"Processing delete event for file: {file_name}")
                    from services.vector_store_ops import delete_from_vector_store
                    await delete_from_vector_store(file_name)
                    await self.storage_manager.delete_status(file_name)
                    logger.info(f"Successfully processed delete event for file: {file_name}")
                elif event.event_type == "Microsoft.Storage.BlobCreated":
                    # Set status to IN_PROGRESS at the beginning of processing
                    await self.storage_manager.update_status(file_name, ProcessingStatus.IN_PROGRESS)
                    logger.info(f"Updated status for {file_name} to IN_PROGRESS")
                    
                    try:
                        from services.vector_store_ops import delete_from_vector_store
                        await delete_from_vector_store(file_name)
                        logger.info(f"Cleaned up existing chunks for {file_name}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up existing chunks: {str(e)}")
                    
                    try:
                        logger.info(f"Downloading file: {file_name}")
                        temp_file_path = await self.storage_manager.download_file(file_name)
                        logger.info(f"Successfully downloaded file: {file_name}")
                        container_client = self.storage_manager.blob_service_client.get_container_client(BLOB_CONTAINER)
                        blob_client = container_client.get_blob_client(blob=file_name)
                        blob_properties = blob_client.get_blob_properties()
                        
                        # Log all metadata for debugging
                        logger.info(f"All metadata for {file_name}: {blob_properties.metadata}")
                        
                        customer_name = blob_properties.metadata.get("customer", "unknown")
                        logger.info(f"Retrieved customer name from metadata: {customer_name}")
                        
                        # Process file based on extension
                        elements = self._partition_file(temp_file_path, file_name)
                        
                        # Create chunks from elements
                        chunks = chunk_by_title(elements, max_characters=5000, new_after_n_chars=6000)
                        logger.info(f"Created {len(chunks)} chunks from {file_name}")
                        
                        # Create contextualized chunks
                        full_document = " ".join([chunk.text for chunk in chunks if hasattr(chunk, 'text') and chunk.text.strip()])
                        logger.info(f"Generating contextualized chunks for {file_name}")
                        contextualized_chunks = await self.contextualizer.contextualize_chunks(full_document, [chunk.text for chunk in chunks])
                        logger.info(f"Created {len(contextualized_chunks)} contextualized chunks for {file_name}")
                        
                        # Create documents for indexing
                        documents = self._create_documents(contextualized_chunks, file_name, customer_name, chunks)
                        
                        # Index documents
                        from services.vector_store_ops import index_documents
                        await index_documents(documents)
                        
                        # Update status to COMPLETED
                        await self.storage_manager.update_status(file_name, ProcessingStatus.COMPLETED)
                        logger.info(f"Completed processing file: {file_name}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_name}: {str(e)}")
                        await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
                        raise
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
                await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
                raise
            finally:
                try:
                    if 'temp_file_path' in locals() and temp_file_path:
                        os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temp file: {str(e)}")
    
    def _partition_file(self, file_path, file_name):
        """Partition a file based on its extension."""
        file_extension = os.path.splitext(file_name)[1].lower()
        logger.info(f"Processing {file_extension} file: {file_name}")
        
        if file_extension == '.pdf':
            elements = partition_pdf(
                filename=file_path,
                strategy="fast",
                include_metadata=True
            )
            logger.info(f"PDF Partitioning - Number of elements: {len(elements)}")
        elif file_extension in ['.doc', '.docx']:
            elements = partition_docx(filename=file_path)
            logger.info(f"DOCX Partitioning - Number of elements: {len(elements)}")
        elif file_extension in ['.xlsx', '.xls']:
            elements = partition_xlsx(filename=file_path)
            elements = process_excel_elements(elements)  # Process Excel elements
            logger.info(f"XLSX Partitioning - Number of elements: {len(elements)}")
        else:
            supported_formats = ['.pdf', '.doc', '.docx', '.xlsx', '.xls']
            raise ValueError(f"Unsupported file type: {file_extension}. Supported formats are: {', '.join(supported_formats)}")
        
        return elements
    
    def _create_documents(self, contextualized_chunks, file_name, customer_name, original_chunks):
        """Create documents for indexing."""
        documents = []
        base_id = sanitize_id(file_name)
        logger.info(f"Converting contextualized chunks to documents for {file_name}")
        
        for i, chunk in enumerate(contextualized_chunks):
            if chunk.strip():
                metadata = {
                    "source": file_name,
                    "chunk_number": i + 1,
                    "customer": customer_name
                }
                
                # Add file type to metadata
                file_extension = os.path.splitext(file_name)[1].lower()
                metadata["file_type"] = file_extension.lstrip('.')
                
                # For Excel files, try to extract and include sheet information
                if file_extension in ['.xlsx', '.xls'] and i < len(original_chunks) and hasattr(original_chunks[i], 'excel_html'):
                    metadata["excel_html"] = original_chunks[i].excel_html
                
                # Generate a unique ID for this chunk
                chunk_id = f"{base_id}_{i+1}"
                metadata["id"] = chunk_id
                metadata["last_update"] = datetime.utcnow().isoformat()
                
                # Create document
                document = DocumentIn(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(document)
        
        logger.info(f"Created {len(documents)} documents for {file_name}")
        return documents