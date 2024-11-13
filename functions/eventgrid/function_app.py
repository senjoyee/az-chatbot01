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
    """Process all documents from blob storage whenever any blob event occurs."""
    try:
        # Get any loader (we'll use it to load all documents)
        loader = DocumentLoaderFactory.get_loader(
            file_extension='.pdf',  # Any supported extension works here
            conn_string=BLOB_CONN_STRING,
            container_name=CONTAINER_NAME
        )
        
        # Load all documents from container - this handles both new files and deletions
        # through the cleanup="full" parameter in the indexing API
        documents = loader.load_from_container()
        chunk_data = []
        
        if documents:
            # Process chunks with consistent metadata
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Prepare chunks for indexing
            for i, chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue
                    
                chunk_data.append({
                    'page_content': chunk.page_content,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                })

        # Always send to FastAPI endpoint - even if chunk_data is empty
        # The cleanup="full" parameter will handle cleanup of old documents
        response = requests.post(
            FASTAPI_ENDPOINT,
            json=chunk_data
        )
        
        if not response.ok:
            raise Exception(f"FastAPI endpoint returned error: {response.status_code} - {response.text}")

        logging.info(f"Successfully processed {len(chunk_data)} chunks from container")
            
    except Exception as e:
        logging.error(f"Error processing event: {str(e)}")
        logging.error(f"Event data: {event.get_json()}")
        raise
