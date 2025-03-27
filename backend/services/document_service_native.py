"""
Document service implementation using native Azure AI Search SDK.
This replaces the langchain-based implementation with direct SDK calls.
"""

import logging
import json
from typing import List, Dict, Any
from fastapi import HTTPException

from config.logging_config import setup_logging
from config.azure_search_native import azure_search_client, embeddings_native
from models.schemas import DocumentIn

logger = setup_logging()

class DocumentServiceNative:
    """
    Document service implementation using native Azure AI Search SDK.
    Provides methods for indexing, retrieving, and deleting documents.
    """
    
    def __init__(self):
        self.azure_search_client = azure_search_client
        self.embeddings = embeddings_native

    async def delete_from_vector_store(self, filename: str) -> dict:
        """
        Delete all document chunks associated with a specific file.
        
        Args:
            filename: The source filename to delete chunks for
            
        Returns:
            dict: A message indicating the number of chunks deleted
        """
        try:
            logger.info(f"Starting deletion process for file: {filename}")
            
            # Escape single quotes by doubling them for OData filter
            escaped_source = filename.replace("'", "''")
            
            logger.info(f"Searching for documents with source: {filename}")
            
            # Use the native search client to find documents to delete
            documents_to_delete = self.azure_search_client.search_client.search(
                search_text="*",
                filter=f"source eq '{escaped_source}'"
            )
            
            deleted_count = 0
            for doc in documents_to_delete:
                logger.info(f"Deleting chunk {deleted_count + 1} with ID: {doc['id']}")
                self.azure_search_client.search_client.delete_documents(documents=[{"id": doc["id"]}])
                deleted_count += 1
                
            logger.info(f"Deletion complete. Removed {deleted_count} chunks for file: {filename}")
            return {
                "message": f"Deleted {deleted_count} chunks for file {filename}",
                "deleted_count": deleted_count
            }
        except Exception as e:
            logger.error(f"Error deleting chunks for file {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def index_documents(self, documents_in: List[DocumentIn]):
        """
        Index a list of documents in Azure AI Search.
        
        Args:
            documents_in: List of DocumentIn objects to index
            
        Returns:
            dict: A message indicating the number of documents indexed
        """
        logger.info(f"Starting indexing process for {len(documents_in)} documents")
        try:
            documents_to_index = []
            
            for document in documents_in:
                try:
                    # Log the document metadata for debugging
                    logger.info(f"Processing document for indexing: ID={document.metadata.get('id', 'unknown')}, "
                              f"Source={document.metadata.get('source', 'unknown')}, "
                              f"Customer={document.metadata.get('customer', 'unknown')}")
                    
                    # Generate embedding using the native embeddings service
                    embedding = self.embeddings.embed_query(document.page_content)
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    continue
                    
                is_contextualized = "\n\n" in document.page_content
                
                # Create the document object for indexing
                document_obj = {
                    "id": document.metadata["id"],
                    "content": document.page_content,
                    "content_vector": embedding,
                    "metadata": json.dumps(document.metadata),
                    "source": document.metadata.get("source", "unknown"),
                    "customer": document.metadata.get("customer", "unknown").lower(),
                    "last_update": document.metadata["last_update"],
                    "contextualized": is_contextualized
                }
                
                # Log the final document object for debugging
                logger.info(f"Document ready for indexing: ID={document_obj['id']}, "
                          f"Customer={document_obj['customer']}")
                
                documents_to_index.append(document_obj)
                
            if documents_to_index:
                # Use the native search client to upload documents
                result = self.azure_search_client.search_client.upload_documents(documents=documents_to_index)
                logger.info(f"Successfully indexed {len(documents_to_index)} documents")
            else:
                logger.warning("No documents to index.")
                
        except Exception as e:
            logger.error(f"Error during document indexing: {str(e)}")
            raise
            
        logger.info(f"Completed indexing process. Successfully indexed {len(documents_in)} documents")
        return {"message": f"Successfully indexed {len(documents_in)} documents"}
