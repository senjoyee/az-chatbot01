import logging
import os
import json
import datetime
from urllib.parse import urlparse
import requests
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure_blob_utils.base_loader import DocumentLoaderFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

app = func.FunctionApp()

# Environment variables with validation
def get_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is not set.")
    return value

FASTAPI_ENDPOINT = get_env_var("FASTAPI_INDEX_DOCUMENTS_ENDPOINT")
BLOB_CONN_STRING = get_env_var("BLOB_CONN_STRING")
CONTAINER_NAME = get_env_var("BLOB_CONTAINER")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

# Add new constant for supported file types
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF Document',
    '.doc': 'Word Document',
    '.docx': 'Word Document',
    '.txt': 'Text Document',
    '.md': 'Markdown Document',
    '.json': 'JSON Document',
    '.csv': 'CSV Document',
    '.log': 'Log Document'
}

def get_effective_extension(blob_name: str) -> str:
    """Determine the effective file extension, using the last extension in the filename.
    
    Args:
        blob_name (str): Name of the blob/file
        
    Returns:
        str: The effective file extension (e.g., '.pdf', '.docx')
    """
    # Split all extensions and get the last one
    parts = blob_name.lower().split('.')
    if len(parts) > 1:
        last_extension = f".{parts[-1]}"
        
        # Check if the last extension is supported
        if last_extension in SUPPORTED_EXTENSIONS:
            logging.info(f"Using last extension '{last_extension}' from filename: {blob_name}")
            return last_extension
            
        # If last extension isn't supported, try to find a supported one from right to left
        for i in range(len(parts)-1, 0, -1):
            ext = f".{parts[i]}"
            if ext in SUPPORTED_EXTENSIONS:
                logging.info(f"Found supported extension '{ext}' in filename: {blob_name}")
                return ext
    
    # Default to the simple splitext behavior as fallback
    extension = os.path.splitext(blob_name)[1].lower()
    logging.info(f"Using fallback extension '{extension}' for filename: {blob_name}")
    return extension

@app.function_name(name="blobtrigger03")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
    """Process document uploads and deletions from blob storage via EventGrid trigger."""
    try:
        event_data = event.get_json()
        event_type = event.event_type  # Get the event type
        
        # Handle different event types
        if event_type == "Microsoft.Storage.BlobDeleted":
            # Get blob URL from the correct path in event data
            url = event_data.get('url', '')  # Direct access to url field
            if not url:
                logging.error(f"Event data structure: {event_data}")
                raise ValueError("No blob URL found in deletion event data")
                
            parsed_url = urlparse(url)
            blob_name = parsed_url.path.split('/')[-1]
            source_id = f"{CONTAINER_NAME}/{blob_name}"
            
            logging.info(f"Processing deletion for blob: {blob_name}")
            
            # Send empty document list with same source_id to trigger cleanup
            response = requests.post(
                FASTAPI_ENDPOINT,
                json=[{
                    'page_content': '',
                    'metadata': {
                        'source': source_id,
                        'is_deleted': True
                    }
                }]
            )
            
            if not response.ok:
                raise Exception(f"FastAPI endpoint returned error on deletion: {response.status_code} - {response.text}")
                
            logging.info(f"Successfully processed deletion for {blob_name}")
            return
            
        elif event_type == "Microsoft.Storage.BlobCreated":
            try:
                # Get blob URL from the correct path in event data
                url = event_data.get('url', '')  # Direct access to url field
                if not url:
                    logging.error(f"Event data structure: {event_data}")
                    raise ValueError("No blob URL found in event data")

                # Parse blob URL to get blob name
                parsed_url = urlparse(url)
                blob_name = parsed_url.path.split('/')[-1]

                # Get effective file extension
                file_extension = get_effective_extension(blob_name)
                logging.info(f"Detected file extension '{file_extension}' for file: {blob_name}")
                
                # Validate file type
                if file_extension not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                    
                doc_type = SUPPORTED_EXTENSIONS[file_extension]
                logging.info(f"Processing {doc_type}: {blob_name}")

                # Get appropriate loader and process document
                loader = DocumentLoaderFactory.get_loader(
                    file_extension=file_extension,
                    conn_string=BLOB_CONN_STRING,
                    container_name=CONTAINER_NAME
                )
                
                try:
                    documents = loader.load(blob_name)
                    logging.info(f"Successfully loaded {doc_type} with {len(documents)} documents")
                except Exception as e:
                    logging.error(f"Error loading {doc_type} {blob_name}: {str(e)}")
                    raise
                
                # Process chunks with consistent metadata
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len
                )
                
                chunks = text_splitter.split_documents(documents)
                
                # Prepare chunks while preserving source information
                chunk_data = []
                source_id = f"{CONTAINER_NAME}/{blob_name}"  # Consistent with loader's source ID
                
                for i, chunk in enumerate(chunks):
                    if not chunk.page_content.strip():
                        logging.warning(f"Skipping empty chunk {i} from {blob_name}")
                        continue
                        
                    # Preserve original metadata and add chunk-specific info
                    chunk_metadata = {
                        **chunk.metadata,  # Original metadata from loader
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'source': source_id,  # Ensure source ID is consistent
                    }
                        
                    chunk_data.append({
                        'page_content': chunk.page_content,
                        'metadata': chunk_metadata
                    })

                if not chunk_data:
                    raise ValueError(f"No valid content chunks extracted from {blob_name}")

                # Send to FastAPI endpoint
                response = requests.post(
                    FASTAPI_ENDPOINT,
                    json=chunk_data
                )
                
                if not response.ok:
                    raise Exception(f"FastAPI endpoint returned error: {response.status_code} - {response.text}")

                logging.info(f"Successfully processed {len(chunk_data)} chunks from {doc_type} {blob_name}")
                
            except Exception as e:
                logging.error(f"Error processing event: {str(e)}")
                logging.error(f"Event data: {event.get_json()}")
                logging.error(f"Event type: {event.event_type}")
                raise
            
        else:
            logging.info(f"Ignoring unsupported event type: {event_type}")
            return
            
    except Exception as e:
        logging.error(f"Error processing event: {str(e)}")
        logging.error(f"Event data: {event.get_json()}")
        logging.error(f"Event type: {event.event_type}")
        raise
