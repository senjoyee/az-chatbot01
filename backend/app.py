import os
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import json
import asyncio
from datetime import datetime

# Import utility functions
from utils.helpers import escape_odata_filter_value, sanitize_id, serialize_metadata, extract_source
from config.logging_config import setup_logging
from config.settings import (BLOB_CONN_STRING, BLOB_CONTAINER)
from config.azure_search import search_client, embeddings
from models.schemas import ConversationRequest, DocumentIn, BlobEvent, FileProcessingStatus
from models.enums import ProcessingStatus
from services.agent import run_agent
from services.contextualizer import Contextualizer
from azure_storage import AzureStorageManager

# Setup logging using the configuration module
logger = setup_logging()

# Initialize storage services
storage_manager = AzureStorageManager()
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

contextualizer = Contextualizer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://documentchatbot01.azurewebsites.net",  # Frontend URL
        "https://jsragfunc01.azurewebsites.net",          # Function App URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global processing lock to enforce sequential processing for file indexing
processing_lock = asyncio.Lock()

@app.get("/file_status/{file_name}")
async def get_file_status(file_name: str) -> FileProcessingStatus:
    """Get the processing status of a file"""
    status, error_message = await storage_manager.get_status(file_name)
    return FileProcessingStatus(
        status=status,
        file_name=file_name,
        error_message=error_message
    )

@app.get("/file_statuses")
async def get_file_statuses(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[ProcessingStatus] = None
):
    """
    Get all file processing statuses with pagination and optional status filtering.
    """
    try:
        all_statuses = await storage_manager.get_all_statuses(status_filter)
        # Sort by last_updated in descending order
        all_statuses.sort(key=lambda x: x["last_updated"], reverse=True)
        # Calculate pagination
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
        logger.error(f"Error getting file statuses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation")
async def ask_question(request: ConversationRequest) -> dict:
    try:
        question = request.question
        conversation = request.conversation

        logger.info(f"Received question: {question}")
        logger.info(f"Received conversation: {conversation}")

        # Run the Langgraph agent
        result = await run_agent(question, conversation.conversation)
        answer = result.get("response")  # Accessing the 'response' from the dict

        return {
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/listfiles")
async def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        blob_list = list(container_client.list_blobs())
        # Calculate pagination
        start = (page - 1) * page_size
        end = start + page_size
        files = [{
            "name": blob.name,
            "size": blob.size,
            "lastModified": blob.last_modified.isoformat(),
            "contentType": blob.content_settings.content_type
        } for blob in blob_list]
        return {
            "total_files": len(files),
            "files": files[start:end],
            "page": page,
            "total_pages": (len(files) - 1) // page_size + 1,
        }
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/deletefile/{filename}")
async def delete_file(filename: str):
    logger.info(f"Attempting to delete file: {filename}")
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
    blob_client = container_client.get_blob_client(blob=filename)

    try:
        # Check if blob exists first
        if not blob_client.exists():
            logger.error(f"File {filename} not found")
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        # First delete from vector store to maintain data consistency
        logger.info(f"Deleting chunks from vector store for file: {filename}")
        deletion_result = await delete_from_vector_store(filename)
        logger.info(f"Vector store deletion result: {deletion_result}")

        # Then delete the blob
        blob_client.delete_blob()
        logger.info(f"Successfully deleted file: {filename}")

        # Also delete the file status
        await storage_manager.delete_status(filename)
        logger.info(f"Successfully deleted status for file: {filename}")

        return {
            "message": f"File {filename} and its {deletion_result['deleted_count']} chunks deleted successfully",
            "deleted_count": deletion_result['deleted_count']
        }
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadfiles")
async def upload_files(files: list[UploadFile] = File(...),
                       customer_names: list[str] = Form(...)):
    results = []

    # Initialize all files as NOT_STARTED
    for file in files:
        await storage_manager.update_status(file.filename, ProcessingStatus.NOT_STARTED)

    # Process files strictly sequentially
    for file in files:
        try:
            # Mark current file as IN_PROGRESS
            await storage_manager.update_status(file.filename, ProcessingStatus.IN_PROGRESS)
            # Process single file
            result = await process_single_file_async(file, customer_names)
            await storage_manager.update_status(file.filename, ProcessingStatus.COMPLETED)
            results.append({"filename": file.filename, "status": "success"})
        except Exception as e:
            await storage_manager.update_status(file.filename, ProcessingStatus.FAILED)
            results.append({"filename": file.filename, "status": "error", "message": str(e)})

    return {"results": results}

async def process_single_file_async(file: UploadFile, customer_names: list[str]):
    # Add size validation
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
    uploaded_files = []

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/(1024*1024)}MB"
        )

    try:
        blob_client = container_client.get_blob_client(blob=file.filename)
        blob_client.upload_blob(contents,
                                overwrite=True,
                                metadata={"customer": customer_names[0]})
        uploaded_files.append(file.filename)
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")

    return {"uploaded_files": uploaded_files}

@app.post("/process_uploaded_files")
async def process_uploaded_files(event: BlobEvent):
    file_name = event.file_name

    # Check if we can process this file
    can_process, error_message = await storage_manager.check_and_acquire_processing_lock(BLOB_CONTAINER, file_name)
    if not can_process:
        return FileProcessingStatus(
            status=ProcessingStatus.IN_PROGRESS,
            file_name=file_name,
            error_message=error_message
        )

    try:
        # Start processing in background task
        asyncio.create_task(process_file_async(event))
        return FileProcessingStatus(
            status=ProcessingStatus.IN_PROGRESS,
            file_name=file_name,
            start_time=datetime.utcnow()
        )
    except Exception as e:
        # Update status on error
        await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_async(event: BlobEvent):
    """Asynchronous file processing function, now enforced to run sequentially using a global lock"""
    async with processing_lock:
        file_name = event.file_name
        logger.info(f"Starting processing for file: {file_name}")

        try:
            if event.event_type == "Microsoft.Storage.BlobDeleted":
                logger.info(f"Processing delete event for file: {file_name}")
                await delete_from_vector_store(file_name)
                await storage_manager.update_status(file_name, ProcessingStatus.COMPLETED)
                logger.info(f"Successfully processed delete event for file: {file_name}")

            elif event.event_type == "Microsoft.Storage.BlobCreated":
                # Clean up existing chunks if necessary
                try:
                    await delete_from_vector_store(file_name)
                    logger.info(f"Cleaned up existing chunks for {file_name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up existing chunks: {str(e)}")

                try:
                    # Download and process the file
                    logger.info(f"Downloading file: {file_name}")
                    temp_file_path = await storage_manager.download_file(file_name)
                    logger.info(f"Successfully downloaded file: {file_name}")

                    # Get customer from blob metadata
                    container_client = storage_manager.blob_service_client.get_container_client(BLOB_CONTAINER)
                    blob_client = container_client.get_blob_client(blob=file_name)
                    blob_properties = blob_client.get_blob_properties()
                    customer_name = blob_properties.metadata.get("customer", "unknown")
                    logger.info(f"Retrieved customer name from metadata: {customer_name}")

                    # Process the file based on its file extension
                    file_extension = os.path.splitext(file_name)[1].lower()
                    logger.info(f"Processing {file_extension} file: {file_name}")

                    if file_extension == '.pdf':
                        elements = partition_pdf(
                            filename=temp_file_path,
                            strategy="hi_res",
                            infer_table_structure=True,
                            include_metadata=True
                        )
                        logger.info(f"PDF Partitioning - Number of elements: {len(elements)}")
                    elif file_extension in ['.doc', '.docx']:
                        elements = partition_docx(
                            filename=temp_file_path
                        )
                        logger.info(f"DOCX Partitioning - Number of elements: {len(elements)}")
                    else:
                        raise ValueError(f"Unsupported file type: {file_extension}")

                    # Chunk the document using title-based chunking
                    chunks = chunk_by_title(
                        elements,
                        max_characters=15000,
                        new_after_n_chars=16000,
                    )
                    logger.info(f"Created {len(chunks)} chunks from {file_name}")

                    full_document = " ".join([chunk.text for chunk in chunks if hasattr(chunk, 'text') and chunk.text.strip()])

                    # Generate contextualized chunks asynchronously
                    logger.info(f"Generating contextualized chunks for {file_name}")
                    contextualized_chunks = await contextualizer.contextualize_chunks(full_document, [chunk.text for chunk in chunks])
                    logger.info(f"Created {len(contextualized_chunks)} contextualized chunks for {file_name}")

                    # Convert contextualized chunks to DocumentIn objects
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
                            doc = DocumentIn(
                                page_content=chunk,
                                metadata={
                                    **serialize_metadata(metadata),
                                    "last_update": datetime.utcnow().isoformat() + "Z",
                                    "id": f"{base_id}_{i:04d}"
                                }
                            )
                            documents.append(doc)

                    logger.info(f"Created {len(documents)} documents from contextualized chunks for {file_name}")

                    # Index the documents
                    logger.info(f"Starting indexing for {len(documents)} contextualized documents from {file_name}")
                    await index_documents(documents)
                    logger.info(f"Successfully indexed {len(documents)} contextualized documents from {file_name}")

                    # Update processing status
                    await storage_manager.update_status(file_name, ProcessingStatus.COMPLETED)
                    logger.info(f"Completed processing file: {file_name}")

                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")
                    await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
                    raise
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
            raise
        finally:
            try:
                if 'temp_file_path' in locals() and temp_file_path:
                    os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temp file: {str(e)}")

async def delete_from_vector_store(filename: str) -> dict:
    """
    Delete all document chunks associated with a specific file from the vector store.
    """
    try:
        logger.info(f"Starting deletion process for file: {filename}")
        # Escape filename for OData filter
        source = filename
        escaped_source = escape_odata_filter_value(source)
        logger.info(f"Searching for documents with source: {source}")

        # Search for documents using the source field
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

@app.post("/index_documents/")
async def index_documents(documents_in: List[DocumentIn]):
    """Index documents into Azure Search."""
    logger.info(f"Starting indexing process for {len(documents_in)} documents")
    try:
        # Prepare documents for indexing
        documents_to_index = []
        for document in documents_in:
            try:
                embedding = embeddings.embed_query(document.page_content)
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                continue

            # Determine if the chunk is contextualized based on the presence of context
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
            documents_to_index.append(document_obj)

        # Upload documents to Azure Search
        if documents_to_index:
            result = search_client.upload_documents(documents=documents_to_index)
            logger.info(f"Successfully indexed {len(documents_to_index)} documents")
        else:
            logger.warning("No documents to index.")
    except Exception as e:
        logger.error(f"Error during document indexing: {str(e)}")
        raise
    logger.info(f"Completed indexing process. Successfully indexed {len(documents_in)} documents")
    return {"message": f"Successfully indexed {len(documents_in)} documents"}

@app.post("/reset_file_status/{filename}")
async def reset_file_status(filename: str):
    """Reset the processing status of a file to NOT_STARTED."""
    try:
        await storage_manager.reset_status(filename)
        return {"message": f"Successfully reset status for {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))