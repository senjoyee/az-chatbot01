import logging  
import os  
import time  
import requests  
import azure.functions as func  
from requests.adapters import HTTPAdapter  
from requests.packages.urllib3.util.retry import Retry  

app = func.FunctionApp()  

FASTAPI_ENDPOINT = os.getenv("FASTAPI_PROCESS_FILES_ENDPOINT")  
MAX_STATUS_CHECKS = 3  # Number of times to check status before succeeding  
INITIAL_WAIT = 10  
MAX_WAIT = 60  
STATUS_CHECK_TIMEOUT = 30  

if not FASTAPI_ENDPOINT:  
    raise ValueError("FASTAPI_PROCESS_FILES_ENDPOINT environment variable is not set.")  

# Configure retry strategy for connection issues  
retry_strategy = Retry(  
    total=3,  
    backoff_factor=1,  
    status_forcelist=[500, 502, 503, 504]  
)  
http = requests.Session()  
http.mount("https://", HTTPAdapter(max_retries=retry_strategy))  

def get_status_url(file_name: str) -> str:  
    """Get the status check URL for a file"""  
    base_url = FASTAPI_ENDPOINT.rsplit('/', 1)[0]  
    return f"{base_url}/file_status/{file_name}"  

@app.function_name(name="blobtrigger01")  
@app.event_grid_trigger(arg_name="event")  
def eventGridTest(event: func.EventGridEvent):  
    try:  
        # Extract event type and file path from the event  
        event_type = event.event_type  
        file_path = event.subject  
        file_name = file_path.split('/')[-1]  

        logging.info(f"Event Type: {event_type}")  
        logging.info(f"File Name: {file_name}")  

        # First check if the file is already being processed  
        try:  
            status_response = http.get(get_status_url(file_name), timeout=STATUS_CHECK_TIMEOUT)  
            status_response.raise_for_status()  
            status_data = status_response.json()  

            if status_data["status"] == "in_progress":  
                logging.info(f"File {file_name} is already being processed. Completing function successfully.")  
                return  
            elif status_data["status"] == "completed":  
                logging.info(f"File {file_name} has been re-uploaded. Starting reprocessing.")  
                # Proceed to initiate processing  
            elif status_data["status"] == "failed":  
                logging.info(f"Previous processing attempt failed. Starting new processing.")  

        except requests.RequestException as e:  
            # If we can't check status (e.g., 404), proceed with processing  
            logging.info(f"Could not check status, proceeding with processing: {str(e)}")  
            # You may decide to reset the status here if necessary  

        # Reset processing status to 'not_started'  
        try:  
            reset_status_response = http.post(  
                f"{FASTAPI_ENDPOINT.rsplit('/', 1)[0]}/reset_file_status/{file_name}",  
                timeout=30  
            )  
            reset_status_response.raise_for_status()  
            logging.info(f"Reset status for file {file_name} to 'not_started'")  
        except requests.RequestException as e:  
            logging.error(f"Failed to reset status for {file_name}: {str(e)}")  
            # Decide whether to proceed or not based on your requirements  

        # Initial request to start processing  
        logging.info(f"Making request to {FASTAPI_ENDPOINT} to process documents")  
        try:  
            response = http.post(  
                FASTAPI_ENDPOINT,  
                json={  
                    "event_type": event_type,  
                    "file_name": file_name  
                },  
                timeout=30  
            )  
            response.raise_for_status()  
            initial_status = response.json()  

            # If the response indicates the file is already being processed, we're done  
            if initial_status.get("status") == "in_progress":  
                logging.info(f"File {file_name} is already being processed by another instance. Completing function successfully.")  
                return  

        except requests.RequestException as e:  
            if e.response and e.response.status_code == 409:  # Conflict - already processing  
                logging.info(f"File {file_name} is already being processed (409 response). Completing function successfully.")  
                return  
            raise  # Re-raise other errors  

        # If it's a delete operation, we're done  
        if event_type == "Microsoft.Storage.BlobDeleted":  
            return  

        # For create/update operations, do a few status checks  
        wait_time = INITIAL_WAIT  
        checks_done = 0  

        while checks_done < MAX_STATUS_CHECKS:  
            try:  
                time.sleep(wait_time)  
                status_response = http.get(get_status_url(file_name), timeout=STATUS_CHECK_TIMEOUT)  
                status_response.raise_for_status()  
                status_data = status_response.json()  

                if status_data["status"] == "completed":  
                    logging.info(f"File {file_name} processed successfully")  
                    return  
                elif status_data["status"] == "failed":  
                    # Log the error but return successfully to prevent Event Grid retries  
                    error_msg = status_data.get('error_message', 'Unknown error')  
                    logging.error(f"File processing failed: {error_msg}")  
                    return  
                elif status_data["status"] == "in_progress":  
                    logging.info(f"Processing in progress for {file_name}")  
                    wait_time = min(wait_time * 1.5, MAX_WAIT)  
                    checks_done += 1  
                else:  
                    logging.warning(f"Unexpected status: {status_data['status']}")  
                    checks_done += 1  

            except requests.RequestException as e:  
                logging.warning(f"Error checking status: {str(e)}")  
                checks_done += 1  

        # If we get here, processing is still ongoing but we'll complete successfully  
        logging.info(f"Processing is still ongoing for {file_name}. Completing function successfully.")  
        return  

    except Exception as e:  
        logging.error(f"Error processing event: {str(e)}")  
        logging.error(f"Full error details: {repr(e)}")  
        # Complete successfully even on errors to prevent Event Grid retries  
        return  