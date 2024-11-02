import logging
import os
import requests
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = func.FunctionApp()

FASTAPI_ENDPOINT = os.getenv("FASTAPI_INDEX_DOCUMENTS_ENDPOINT")
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

if not FASTAPI_ENDPOINT:
    raise ValueError(
        "FASTAPI_INDEX_DOCUMENTS_ENDPOINT environment variable is not set."
    )

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

@app.function_name(name="blobtrigger02")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
    loader = AzureBlobStorageContainerLoader(
        conn_str=BLOB_CONN_STRING, 
        container=CONTAINER_NAME
    )
    data = loader.load()
    
    # Add debug logging for initial document
    for idx, doc in enumerate(data):
        logging.info(f"Original document {idx} content length: {len(doc.page_content)}")
        logging.info(f"First 100 characters: {doc.page_content[:100]}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Add debug logging for chunking process
    logging.info(f"Attempting to split with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    split_documents = text_splitter.split_documents(data)
    
    logging.info(f"Document count after splitting: {len(split_documents)}")
    if len(split_documents) > 0:
        logging.info(f"First chunk size: {len(split_documents[0].page_content)}")
    else:
        logging.warning("No chunks were created! Checking document content...")
        for doc in data:
            if not doc.page_content.strip():
                logging.warning("Document content is empty or only whitespace!")
            
    documents_in = [
        {
            "page_content": doc.page_content,
            "metadata": {"source": doc.metadata["source"]},
        }
        for doc in split_documents
    ]
    response = requests.post(FASTAPI_ENDPOINT, json=documents_in)

    if response.status_code == 200:
        logging.info("Documents sent successfully to FastAPI endpoint.")
    else:
        logging.error(
            f"Failed to send documents. Status Code: {response.status_code} Response: {response.text}"
        )
