import logging
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from azure.storage.blob import BlobServiceClient
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title
from fastapi import HTTPException

from config.logging_config import setup_logging
from config.settings import BLOB_CONN_STRING, BLOB_CONTAINER
from config.azure_search import search_client, embeddings
from models.enums import ProcessingStatus
from models.schemas import DocumentIn, BlobEvent
from services.azure_storage import AzureStorageManager
from services.contextualizer import Contextualizer
from services.document_service import DocumentService
from utils.helpers import sanitize_id, serialize_metadata, process_excel_elements, escape_odata_filter_value
import json

logger = setup_logging()

class FileProcessor:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)
        self.contextualizer = Contextualizer()
        self.document_service = DocumentService()
        self.processing_lock = asyncio.Lock()

    async def process_file_async(self, event: BlobEvent):
        async with self.processing_lock:
            file_name = event.file_name
            logger.info(f"Starting to process file: {file_name}")
            try:
                if event.event_type == "Microsoft.Storage.BlobDeleted":
                    logger.info(f"Processing delete event for file: {file_name}")
                    await self.document_service.delete_from_vector_store(file_name)
                    await self.storage_manager.delete_status(file_name)
                    logger.info(f"Successfully processed delete event for file: {file_name}")
                elif event.event_type == "Microsoft.Storage.BlobCreated":
                    # Set status to IN_PROGRESS at the beginning of processing
                    await self.storage_manager.update_status(file_name, ProcessingStatus.IN_PROGRESS)
                    logger.info(f"Updated status for {file_name} to IN_PROGRESS")
                    
                    try:
                        await self.document_service.delete_from_vector_store(file_name)
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
                        file_extension = os.path.splitext(file_name)[1].lower()
                        logger.info(f"Processing {file_extension} file: {file_name}")
                        if file_extension == '.pdf':
                            elements = partition_pdf(
                                filename=temp_file_path,
                                #strategy="hi_res",
                                #infer_table_structure=True,
                                strategy="fast",
                                include_metadata=True
                            )
                            logger.info(f"PDF Partitioning - Number of elements: {len(elements)}")
                        elif file_extension in ['.doc', '.docx']:
                            elements = partition_docx(filename=temp_file_path)
                            logger.info(f"DOCX Partitioning - Number of elements: {len(elements)}")
                        elif file_extension in ['.xlsx', '.xls']:
                            elements = partition_xlsx(filename=temp_file_path)
                            elements = process_excel_elements(elements)  # Process Excel elements
                            logger.info(f"XLSX Partitioning - Number of elements: {len(elements)}")
                        elif file_extension in ['.ppt', '.pptx']:
                            elements = partition_pptx(filename=temp_file_path)
                            logger.info(f"PPTX Partitioning - Number of elements: {len(elements)}")
                        else:
                            supported_formats = ['.pdf', '.doc', '.docx', '.xlsx', '.xls', '.ppt', '.pptx']
                            raise ValueError(f"Unsupported file type: {file_extension}. Supported formats are: {', '.join(supported_formats)}")
                        chunks = chunk_by_title(elements, max_characters=15000, new_after_n_chars=16000)
                        logger.info(f"Created {len(chunks)} chunks from {file_name}")
                        full_document = " ".join([chunk.text for chunk in chunks if hasattr(chunk, 'text') and chunk.text.strip()])
                        logger.info(f"Generating contextualized chunks for {file_name}")
                        contextualized_chunks = await self.contextualizer.contextualize_chunks(full_document, [chunk.text for chunk in chunks])
                        logger.info(f"Created {len(contextualized_chunks)} contextualized chunks for {file_name}")
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
                                if file_extension in ['.xlsx', '.xls'] and i < len(chunks) and hasattr(chunks[i], 'excel_html'):
                                    metadata["content_type"] = "table"
                                    # Store a reference to the HTML representation
                                    metadata["has_table_html"] = True
                                    
                                    # If the original chunk has sheet name information, include it
                                    if hasattr(chunks[i], 'metadata') and hasattr(chunks[i].metadata, 'sheet_name'):
                                        metadata["sheet_name"] = chunks[i].metadata.sheet_name
                                
                                logger.info(f"Creating document for chunk {i+1} with metadata: {metadata}")
                                doc = DocumentIn(
                                    page_content=chunk,
                                    metadata={**serialize_metadata(metadata),
                                            "last_update": datetime.utcnow().isoformat() + "Z",
                                            "id": f"{base_id}_{i:04d}"}
                                )
                                documents.append(doc)
                        logger.info(f"Created {len(documents)} documents from contextualized chunks for {file_name}")
                        logger.info(f"Starting indexing for {len(documents)} contextualized documents from {file_name}")
                        await self.document_service.index_documents(documents)
                        logger.info(f"Successfully indexed {len(documents)} contextualized documents from {file_name}")
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