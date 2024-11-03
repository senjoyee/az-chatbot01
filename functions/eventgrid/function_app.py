import logging
import os
import json
import requests
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure_blob_utils.loaders import EnhancedAzureBlobStorageContainerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = func.FunctionApp()

FASTAPI_ENDPOINT = os.getenv("FASTAPI_INDEX_DOCUMENTS_ENDPOINT")
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

if not FASTAPI_ENDPOINT:
    raise ValueError("FASTAPI_INDEX_DOCUMENTS_ENDPOINT environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

@app.function_name(name="blobtrigger03")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
    try:
        # Load and process documents
        loader = EnhancedAzureBlobStorageContainerLoader(
            conn_str=BLOB_CONN_STRING, container=CONTAINER_NAME
        )
        data = loader.load()
        
        # Log document metadata
        for idx, doc in enumerate(data):
            logging.info(f"Document {idx} source: {doc.metadata.get('source', 'unknown')}")
            logging.info(f"Document {idx} content length: {len(doc.page_content)}")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(data)
        logging.info(f"Document count after splitting: {len(split_documents)}")

        # Prepare payload
        documents_in = [
            {
                "page_content": doc.page_content,
                "metadata": {"source": doc.metadata["source"]},
            }
            for doc in split_documents
        ]
        
        # Log request details
        logging.info(f"Sending request to: {FASTAPI_ENDPOINT}")
        logging.info(f"Payload size: {len(json.dumps(documents_in))} bytes")
        
        # Send request with timeout
        response = requests.post(
            FASTAPI_ENDPOINT, 
            json=documents_in,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        logging.info("Documents sent successfully to FastAPI endpoint.")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status: {e.response.status_code}")
            logging.error(f"Response body: {e.response.text}")
        raise
        
    except Exception as e:
        logging.error(f"Function failed: {str(e)}")
        raise
