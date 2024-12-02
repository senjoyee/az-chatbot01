import logging
import os
import requests
import azure.functions as func
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = func.FunctionApp()

FASTAPI_ENDPOINT = os.getenv("FASTAPI_PROCESS_FILES_ENDPOINT")

if not FASTAPI_ENDPOINT:
    raise ValueError("FASTAPI_PROCESS_FILES_ENDPOINT environment variable is not set.")

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=retry_strategy))

@app.function_name(name="myblobtrigger")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
    try:
        # Extract event type and file path from the event
        event_type = event.event_type
        file_path = event.subject
        
        # Get just the file name from the full path
        # The subject format is typically: /blobServices/default/containers/{container-name}/blobs/{blob-name}
        file_name = file_path.split('/')[-1]
        
        logging.info(f"Event Type: {event_type}")
        logging.info(f"File Name: {file_name}")
        
        logging.info(f"Making request to {FASTAPI_ENDPOINT} to process documents")
        
        response = http.post(
            FASTAPI_ENDPOINT,
            json={
                "event_type": event_type,
                "file_name": file_name
            },
            timeout=30
        )
        
        logging.info(f"Response status: {response.status_code}")
        logging.info(f"Response body: {response.text}")
        
        response.raise_for_status()
        logging.info("Successfully triggered document processing")
            
    except Exception as e:
        logging.error(f"Error processing event: {str(e)}")
        logging.error(f"Full error details: {repr(e)}")
        raise