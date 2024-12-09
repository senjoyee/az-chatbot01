import os
import logging
from operator import itemgetter
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, StrOutputParser, format_document
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from azure.storage.blob import BlobServiceClient
from dotenv import find_dotenv, load_dotenv
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import tempfile
import json
import sys
import asyncio
from enum import Enum
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
from azure_storage import AzureStorageManager, ProcessingStatus

# Import utility functions
from utils.helpers import escape_odata_filter_value, sanitize_id, serialize_metadata
from config.logging_config import setup_logging
from config.settings import (
    AZURE_SEARCH_SERVICE,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    BLOB_CONN_STRING,
    BLOB_CONTAINER,
    AZURE_SEARCH_SERVICE_ENDPOINT
)

# Setup logging using the configuration module
logger = setup_logging()

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",dimensions=1536)

embedding_function = embeddings.embed_query


# Define fields for Azure Search
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SearchableField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
        searchable=True,
    ),
    SimpleField(  
        name="last_update",  
        type=SearchFieldDataType.DateTimeOffset,  
        filterable=True,  
        sortable=True,  
    ), 
]


scoring_profile = ScoringProfile(  
    name="content_source_freshness_profile",  
    text_weights=TextWeights(weights={  
        "content": 5,  # Highest weight for content  
        "source": 3    # Lower weight for source  
    }),  
    function_aggregation="sum",  
    functions=[  
        FreshnessScoringFunction(  
            field_name="last_update",  
            boost=100,  
            parameters=FreshnessScoringParameters(boosting_duration="P15D"),  
            interpolation="linear"  
        )  
    ]  
)  

index_name: str = AZURE_SEARCH_INDEX

vector_store: AzureSearch = AzureSearch(  
    azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,  
    azure_search_key=AZURE_SEARCH_KEY,  
    index_name=AZURE_SEARCH_INDEX,  
    embedding_function=embedding_function,  
    fields=fields,  
    scoring_profiles=[scoring_profile],  
    default_scoring_profile="content_source_freshness_profile"  
)  

# Initialize services
storage_manager = AzureStorageManager()
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

# Initialize search client for direct operations if needed
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Check if index exists and create if it doesn't
try:
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
    index_client.get_index(AZURE_SEARCH_INDEX)
    logger.info(f"Index {AZURE_SEARCH_INDEX} already exists")
except Exception as e:
    logger.info(f"Index {AZURE_SEARCH_INDEX} does not exist. Creating...")
    logger.info("Index will be created automatically by Langchain")

# Import models from the new module
from models.schemas import Message, Conversation, ConversationRequest, DocumentIn, BlobEvent, FileProcessingStatus
from models.enums import ProcessingStatus

# Initialize the chat model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3
)

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

answer_template = """Please provide an answer based strictly on the following context:
<context>
{context}
</context>

Guidelines for answering:
1. I will analyze the question and context thoroughly
2. Present information in a clear, structured format
3. Use numbered points for main ideas
4. Use a, b for sub-points instead of dashes 
5. Ensure all points are supported by the context
6. Clearly indicate if information is incomplete
7. Maintain consistency throughout the response

Question: {question}

Answer:
[If sufficient information exists in context]
1. [First main point]
    a. [Supporting detail]
    b. [Supporting detail]
2. [Second main point]
    a. [Supporting detail]
    etc.

[If information is incomplete or missing]
Based on the provided context, here is the relevant information that is available:
1. [Available information point]
    a. [Available detail]
2. [Available information point]
    a. [Available detail]

[If no relevant information exists]
The provided context does not contain information to answer this question.
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    logger.info(f"Docstrings: {doc_strings}")
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history):
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": lambda x: _combine_documents(
        vector_store.hybrid_search(
            x["standalone_question"],
            k=10
        )
    ),
    "question": lambda x: x["standalone_question"],
}

# Combine the chain components
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://documentchatbot01.azurewebsites.net",  # Frontend URL
        "https://jsragfunc01.azurewebsites.net",        # Function App URL
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],  # Specify allowed methods
    allow_headers=["*"],
    expose_headers=["*"],
)

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
        # Get all statuses
        all_statuses = await storage_manager.get_all_statuses(status_filter)
        
        # Sort by last_updated in descending order
        all_statuses.sort(key=lambda x: x["last_updated"], reverse=True)
        
        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated results
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
      
      # Debug logging
      logger.info(f"Received question: {question}")
      logger.info(f"Received conversation: {conversation}")

      # Use the messages directly without transformation
      chat_history = conversation.conversation
      
      logger.info(f"Using chat history: {chat_history}")
      
      answer = conversational_qa_chain.invoke({
          "question": question,
          "chat_history": chat_history
      })
      
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
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        # Add size validation
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        uploaded_files = []

        for file in files:
            # Validate file size
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/(1024*1024)}MB"
                )
                
            try:
                blob_client = container_client.get_blob_client(blob=file.filename)
                blob_client.upload_blob(contents, overwrite=True)
                uploaded_files.append(file.filename)
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")

        return {"uploaded_files": uploaded_files}
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """Asynchronous file processing function"""
    file_name = event.file_name
    logger.info(f"Starting processing for file: {file_name}")
    
    try:
        if event.event_type == "Microsoft.Storage.BlobDeleted":
            logger.info(f"Processing delete event for file: {file_name}")
            await delete_from_vector_store(file_name)
            await storage_manager.update_status(file_name, ProcessingStatus.COMPLETED)
            logger.info(f"Successfully processed delete event for file: {file_name}")
        
        elif event.event_type == "Microsoft.Storage.BlobCreated":
            # First, check if we need to clean up existing chunks
            try:
                await delete_from_vector_store(file_name)
                logger.info(f"Cleaned up existing chunks for {file_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up existing chunks: {str(e)}")
            
            try:
                # Download the file
                logger.info(f"Downloading file: {file_name}")
                temp_file_path = await storage_manager.download_file(file_name)
                logger.info(f"Successfully downloaded file: {file_name}")
                
                try:
                    # Process the file
                    file_extension = os.path.splitext(file_name)[1].lower()
                    logger.info(f"Processing {file_extension} file: {file_name}")
                    
                    if file_extension == '.pdf':
                        elements = partition_pdf(
                            filename=temp_file_path,
                            strategy="fast"
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
                        max_characters=6000,
                        new_after_n_chars=5000,
                    )
                    logger.info(f"Created {len(chunks)} chunks from {file_name}")
                    
                    # Convert chunks to documents
                    documents = []
                    base_id = sanitize_id(file_name)
                    logger.info(f"Converting chunks to documents for {file_name}")
                    for i, chunk in enumerate(chunks):
                        if hasattr(chunk, 'text') and chunk.text.strip():
                            # Get metadata
                            metadata = {
                                "source": file_name
                            }
                            if hasattr(chunk, 'metadata'):
                                element_metadata = chunk.metadata.__dict__ if hasattr(chunk.metadata, '__dict__') else chunk.metadata
                                metadata.update(element_metadata)
                            
                            # Extract page numbers
                            page_numbers = set()
                            if hasattr(metadata, 'page_number'):
                                page_num = metadata.page_number
                                if page_num is not None:
                                    page_numbers.add(page_num)
                            elif isinstance(metadata, dict) and 'page_number' in metadata:
                                page_num = metadata['page_number']
                                if page_num is not None:
                                    page_numbers.add(page_num)
                            
                            metadata["page_numbers"] = list(sorted(page_numbers)) if page_numbers else None
                            
                            # Remove title field from metadata
                            metadata.pop('title', None)
                            
                            # Serialize metadata to ensure JSON compatibility
                            serialized_metadata = serialize_metadata(metadata)
                            
                            doc = DocumentIn(
                                page_content=chunk.text,
                                metadata={
                                    **serialized_metadata,
                                    "last_update": datetime.utcnow().isoformat() + "Z",
                                    "id": f"{base_id}_{i:04d}"
                                }
                            )
                            documents.append(doc)
                    
                    logger.info(f"Created {len(documents)} documents from chunks for {file_name}")
                    
                    # Index the documents
                    logger.info(f"Starting indexing for {len(documents)} documents from {file_name}")
                    await index_documents(documents)
                    logger.info(f"Successfully indexed {len(documents)} documents from {file_name}")
                    await storage_manager.update_status(file_name, ProcessingStatus.COMPLETED)
                    logger.info(f"Completed processing file: {file_name}")
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")
                    await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
                    raise
                finally:
                    # Clean up temp file
                    if temp_file_path:
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.warning(f"Error cleaning up temp file: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error downloading file {file_name}: {str(e)}")
                await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
                raise
            
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}")
        await storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))

async def delete_from_vector_store(filename: str) -> dict:
    """
    Delete all document chunks associated with a specific file from the vector store.
    """
    try:
        logger.info(f"Starting deletion process for file: {filename}")
        
        # Escape filename for OData filter
        escaped_filename = escape_odata_filter_value(filename)
        logger.info(f"Searching for documents with escaped source: {escaped_filename}")
        
        documents_to_delete = search_client.search(
            search_text="",
            filter=f"source eq '{escaped_filename}'"
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
async def index_documents(documents_in: list[DocumentIn]):
    logger.info(f"Starting indexing process for {len(documents_in)} documents")
    try:
        for i, document in enumerate(documents_in, 1):
            logger.info(f"Processing document {i}/{len(documents_in)}: {document.metadata.get('source', 'unknown')}")
            try:
                logger.info("Generating embedding...")
                embedding = embeddings.embed_query(document.page_content)
                logger.info(f"Generated embedding of size: {len(embedding)}")
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                continue
            
            doc_id = f"doc_{hash(document.page_content)}"
            logger.info(f"Created document ID: {doc_id}")
            
            document_obj = {
                "id": doc_id,
                "content": document.page_content,
                "content_vector": embedding,
                "metadata": json.dumps(document.metadata),
                "source": document.metadata.get("source", "unknown"),
                "last_update": datetime.utcnow().isoformat() + "Z"
            }
            
            logger.info(f"Indexing document into Azure Search: {doc_id}")
            try:
                result = search_client.upload_documents(documents=[document_obj])
                logger.info(f"Successfully indexed document {i}/{len(documents_in)}")
            except Exception as e:
                logger.error(f"Error indexing document {doc_id}: {str(e)}")
                raise

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
