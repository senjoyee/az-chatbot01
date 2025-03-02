import json
import logging
from typing import List
from fastapi import HTTPException

from models.schemas import DocumentIn
from utils.helpers import escape_odata_filter_value
from config.azure_search import search_client, embeddings

logger = logging.getLogger(__name__)

async def delete_from_vector_store(filename: str) -> dict:
    """Delete documents from the vector store based on filename."""
    try:
        logger.info(f"Starting deletion process for file: {filename}")
        source = filename
        escaped_source = escape_odata_filter_value(source)
        logger.info(f"Searching for documents with source: {source}")
        documents_to_delete = search_client.search(
            search_text="*",
            filter=f"source eq '{escaped_source}'"
        )
        deleted_count = 0
        for doc in documents_to_delete:
            logger.info(f"Deleting chunk {deleted_count + 1} with ID: {doc['id']}")
            search_client.delete_documents(documents=[{"id": doc["id"]}])
            deleted_count += 1
        logger.info(f"Deletion complete. Removed {deleted_count} chunks for file: {filename}")
        return {
            "message": f"Deleted {deleted_count} chunks for file {filename}",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error deleting chunks for file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def index_documents(documents_in: List[DocumentIn]):
    """Index documents in the vector store."""
    logger.info(f"Starting indexing process for {len(documents_in)} documents")
    try:
        documents_to_index = []
        for document in documents_in:
            try:
                # Log the document metadata for debugging
                logger.info(f"Processing document for indexing: ID={document.metadata.get('id', 'unknown')}, "
                           f"Source={document.metadata.get('source', 'unknown')}, "
                           f"Customer={document.metadata.get('customer', 'unknown')}")
                
                embedding = embeddings.embed_query(document.page_content)
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                continue
            is_contextualized = "\n\n" in document.page_content
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
            result = search_client.upload_documents(documents=documents_to_index)
            logger.info(f"Successfully indexed {len(documents_to_index)} documents")
        else:
            logger.warning("No documents to index.")
    except Exception as e:
        logger.error(f"Error during document indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info(f"Completed indexing process. Successfully indexed {len(documents_in)} documents")
    return {"message": f"Successfully indexed {len(documents_in)} documents"}