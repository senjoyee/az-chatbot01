import streamlit as st
import os
import base64
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient
from dotenv import load_dotenv
import pandas as pd

# Configure the page with a favicon and custom title (must be first Streamlit command)
st.set_page_config(
    page_title="Azure File Upload Service",
    page_icon="☁️",
    layout="centered"
)

# Load environment variables
load_dotenv()

# Azure Storage settings
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "").strip()

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB in bytes

# Initialize session state
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = set()
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'customer_names' not in st.session_state:
    st.session_state.customer_names = {}

# Helper functions
def format_size(size_in_bytes):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"

def upload_to_azure(file_data, file_name, customer_name):
    """Upload a file to Azure Blob Storage with customer metadata"""
    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get the blob client
        blob_client = container_client.get_blob_client(file_name)
        
        # Upload the file with metadata
        blob_client.upload_blob(file_data, overwrite=True, metadata={"customer": customer_name})
    except ValueError as ve:
        st.error(f"Connection string error: {str(ve)}")
        st.error("Please check if your connection string is correctly formatted")
        raise
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        st.error("Please verify your Azure credentials and permissions")
        raise

def delete_blob(blob_name):
    """Delete a blob from Azure Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        container_client.delete_blob(blob_name)
        return True
    except Exception as e:
        st.error(f"Error deleting {blob_name}: {str(e)}")
        return False

# Custom CSS for styling
st.markdown("""
    <style>
    .delete-btn {
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        text-decoration: none;
        font-size: 0.8rem;
        transition: all 0.2s;
    }
    .delete-btn:hover {
        background-color: #ff4b4b;
        color: white;
    }
    .stButton button {
        width: 100%;
    }
    /* Upload section styling */
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #ccc;
        margin-bottom: 2rem;
        text-align: center;
    }
    .upload-icon {
        font-size: 3rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .upload-text {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .file-limit {
        color: #888;
        font-size: 0.9rem;
    }
    /* Table styling */
    .file-table {
        margin-top: 2rem;
        border: 1px solid #eee;
        border-radius: 5px;
        padding: 1rem;
    }
    .table-header {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with custom styling
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>File Upload Service</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>Securely upload and manage your files in Azure Blob Storage</p>", unsafe_allow_html=True)

# Styled upload section
st.markdown("""
    <div class='upload-section'>
        <div class='upload-icon'>☁️</div>
        <div class='upload-text'>Drag and drop files here</div>
        <div class='file-limit'>Maximum file size: 200MB</div>
    </div>
""", unsafe_allow_html=True)

# File uploader with custom styling
uploaded_files = st.file_uploader(
    "Upload Files",
    type=None,
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    total_size = sum(file.size for file in uploaded_files)
    if total_size > MAX_FILE_SIZE:
        st.error(f"Total size of files exceeds the limit of {format_size(MAX_FILE_SIZE)}")
    else:
        st.write("### File Details")
        
        # Track if all files have customer names
        all_files_have_customers = True
        
        # Show file details and customer input for each file
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                st.write(f"**Size:** {format_size(uploaded_file.size)}")
                st.write(f"**Upload time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Customer name input for this specific file
                file_key = f"customer_name_{uploaded_file.name}"
                customer_name = st.text_input(
                    "Customer Name",
                    key=file_key,
                    value=st.session_state.customer_names.get(uploaded_file.name, ""),
                    help="Enter the customer name for this file"
                )
                
                # Store customer name in session state
                if customer_name.strip():
                    st.session_state.customer_names[uploaded_file.name] = customer_name.strip()
                else:
                    all_files_have_customers = False

        # Only show upload button if all files have customer names
        if all_files_have_customers:
            if st.button("Upload to Azure", type="primary"):
                try:
                    with st.spinner("Uploading files..."):
                        for uploaded_file in uploaded_files:
                            customer_name = st.session_state.customer_names[uploaded_file.name]
                            upload_to_azure(uploaded_file.getvalue(), uploaded_file.name, customer_name)
                    st.success("Files uploaded successfully!")
                    # Clear selected files and customer names after successful upload
                    st.session_state.selected_files = set()
                    st.session_state.customer_names = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during upload: {str(e)}")
        else:
            st.warning("Please enter customer names for all files before uploading")

# Check Azure Storage settings
if not connection_string:
    st.error("AZURE_STORAGE_CONNECTION_STRING is not set in the environment variables")
else:
    # Validate connection string format
    try:
        parts = dict(part.split('=', 1) for part in connection_string.split(';') if '=' in part)
        required_parts = ['DefaultEndpointsProtocol', 'AccountName', 'AccountKey', 'EndpointSuffix']
        missing_parts = [part for part in required_parts if part not in parts]
        if missing_parts:
            st.error(f"Connection string is missing required parts: {', '.join(missing_parts)}")
        else:
            # Validate AccountKey can be base64 decoded
            try:
                base64.b64decode(parts['AccountKey'])
            except Exception as e:
                st.error(f"AccountKey is not valid base64: {str(e)}")
    except Exception as e:
        st.error(f"Connection string format is invalid: {str(e)}")

if not container_name:
    st.error("AZURE_STORAGE_CONTAINER_NAME is not set in the environment variables")

# Display existing files
st.markdown("<div class='file-table'>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #2c3e50; font-size: 1.5rem; margin-bottom: 1rem;'>Uploaded Files</h2>", unsafe_allow_html=True)
try:
    if connection_string and container_name:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            
            # Get list of blobs
            blobs = list(container_client.list_blobs())
            
            if not blobs:
                st.info("No files found in storage.")
            else:
                # Pagination logic
                ITEMS_PER_PAGE = 10
                total_pages = (len(blobs) - 1) // ITEMS_PER_PAGE + 1
                start_idx = st.session_state.current_page * ITEMS_PER_PAGE
                end_idx = start_idx + ITEMS_PER_PAGE
                current_page_blobs = blobs[start_idx:end_idx]
                
                # Create a container with custom width
                with st.container():
                    # Create columns for the table header
                    cols = st.columns([0.6, 2, 1.5, 2, 1])
                    cols[0].write("Select")
                    cols[1].write("**Filename**")
                    cols[2].write("**Size**")
                    cols[3].write("**Last Modified**")
                    cols[4].write("**Action**")
                    
                    # Display current page blobs
                    for blob in current_page_blobs:
                        cols = st.columns([0.6, 2, 1.5, 2, 1])
                        
                        # Checkbox for selection with proper label
                        if cols[0].checkbox(
                            f"Select {blob.name}",
                            key=f"select_{blob.name}",
                            value=blob.name in st.session_state.selected_files,
                            label_visibility="collapsed"
                        ):
                            st.session_state.selected_files.add(blob.name)
                        else:
                            st.session_state.selected_files.discard(blob.name)
                        
                        # File information
                        cols[1].write(blob.name)
                        cols[2].write(format_size(blob.size))
                        cols[3].write(blob.last_modified.strftime("%Y-%m-%d %H:%M:%S"))
                        
                        # Delete button
                        if cols[4].button("🗑️", key=f"delete_{blob.name}"):
                            if delete_blob(blob.name):
                                st.success(f"Deleted {blob.name}")
                                st.rerun()
                    
                    # Pagination controls
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    with col2:
                        if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                            st.session_state.current_page -= 1
                            st.rerun()
                    with col3:
                        if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1):
                            st.session_state.current_page += 1
                            st.rerun()
                    st.caption(f"Page {st.session_state.current_page + 1} of {total_pages}")
                    
                    # Delete selected files button
                    if st.session_state.selected_files:
                        st.markdown("---")
                        delete_col1, delete_col2, delete_col3 = st.columns([1, 2, 1])
                        with delete_col2:
                            if st.button(f"Delete Selected Files ({len(st.session_state.selected_files)})", 
                                       type="primary"):
                                success_count = 0
                                for blob_name in st.session_state.selected_files:
                                    if delete_blob(blob_name):
                                        success_count += 1
                                
                                if success_count > 0:
                                    st.success(f"Successfully deleted {success_count} files")
                                    st.session_state.selected_files = set()
                                    st.session_state.current_page = 0  # Reset to first page after deletion
                                    st.rerun()

        except Exception as e:
            st.error(f"Error accessing blob storage: {str(e)}")
            st.error("Please verify your Azure credentials and permissions")
    else:
        st.warning("Azure Storage credentials not configured. Please set up your environment variables.")
except Exception as e:
    st.error(f"Error listing files: {str(e)}")