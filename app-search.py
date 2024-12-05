import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
import logging
import tempfile
import sys
import json
from unstructured.documents.elements import CompositeElement

# Initialize logger with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create a placeholder for debug information in Streamlit
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

def log_debug(message: str):
    """Log message to logger only."""
    logger.debug(message)

# Load environment variables
load_dotenv()

# Constants and configurations
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
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
        filterable=True,
    ),
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
        searchable=True,
    )
]

def get_vector_store() -> AzureSearch:
    """Get or create Azure Search vector store."""
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embedding_function,
        fields=fields,
    )

def initialize_search_client():
    """Initialize the Azure Search client."""
    try:
        credential = AzureKeyCredential(AZURE_SEARCH_KEY)
        return SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=credential
        )
    except Exception as e:
        logger.error(f"Error initializing search client: {str(e)}")
        raise

class ConcurrencyError(Exception):
    """Exception raised for concurrency conflicts."""
    pass

class ETagManager:
    @staticmethod
    def get_document_etag(search_client, filename):
        """Get the ETag for a specific document."""
        try:
            # Use the lookup API to get the document directly
            doc = search_client.get_document(key=filename)
            if hasattr(doc, '_meta') and hasattr(doc._meta, 'etag'):
                return doc._meta.etag
            return None
        except Exception as e:
            logger.error(f"Error getting ETag: {str(e)}")
            return None

    @staticmethod
    def verify_etag(search_client, filename, expected_etag):
        """Verify if the document's current ETag matches the expected one."""
        if not expected_etag:
            return True  # If no ETag is provided, skip verification
        current_etag = ETagManager.get_document_etag(search_client, filename)
        return current_etag == expected_etag

def get_indexed_files(search_client):
    """Get list of unique source files from the index with their ETags."""
    try:
        # Get all documents
        results = search_client.search(
            search_text="*",
            select=["source"],
            include_total_count=True,
            query_type="simple",
            search_mode="all",
            top=1000  # Set a high limit to get all files
        )
        
        # Extract unique source values and their ETags using a dictionary
        files_dict = {}
        
        # Add debug logging for total count
        total_count = results.get_count()
        logger.info(f"Total documents found: {total_count}")
        
        for result in results:
            logger.info(f"Raw result: {result}")
            if isinstance(result, dict) and 'source' in result:
                source = result['source']
                # Get ETag using the lookup API
                etag = ETagManager.get_document_etag(search_client, source)
                files_dict[source] = etag
                logger.info(f"Found file: {source} with ETag: {etag}")
            else:
                logger.warning(f"Result missing source field: {result}")
        
        # Convert to list of tuples (filename, etag)
        sorted_files = [(k, v) for k, v in sorted(files_dict.items())]
        
        # Add debug logging
        logger.info(f"Total unique files found: {len(sorted_files)}")
        return sorted_files
    except Exception as e:
        logger.error(f"Error getting indexed files: {str(e)}")
        return []

def add_documents_to_index(documents: List[Dict[str, Any]], vector_store: AzureSearch) -> None:
    """Add documents to Azure Search index."""
    try:
        # Extract texts, metadata, and ids from documents
        texts = [doc["content"] for doc in documents]
        metadatas = [json.loads(doc["metadata"]) for doc in documents]
        ids = [doc["id"] for doc in documents]
        sources = [doc["source"] for doc in documents]
        
        # Debug logging for document structure
        log_debug("Document structure samples:")
        for i in range(min(2, len(documents))):
            log_debug(f"Document {i}:")
            log_debug(f"  Text (first 100 chars): {texts[i][:100]}...")
            log_debug(f"  Metadata: {metadatas[i]}")
            log_debug(f"  ID: {ids[i]}")
            log_debug(f"  Source: {sources[i]}")
        
        # Add documents using add_texts
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids, sources=sources)
        log_debug(f"Successfully added {len(texts)} documents to index")
    except Exception as e:
        log_debug(f"Error adding documents to index: {str(e)}")
        raise e

def sanitize_id(filename: str) -> str:
    """
    Sanitize filename to create a valid Azure Search ID.
    Only keeps alphanumeric chars and underscores, replacing others with dashes.
    Also removes file extensions.
    """
    # Remove file extensions (handles multiple extensions like .docx.pdf)
    base_name = filename.split('.')[0]
    # Replace any non-alphanumeric chars (except underscore) with dash
    sanitized = ''.join(c if c.isalnum() or c == '_' else '-' for c in base_name)
    # Remove any duplicate dashes
    sanitized = '-'.join(filter(None, sanitized.split('-')))
    return sanitized

def process_document(file_path: str, original_filename: str, file_type: str) -> List[Dict[str, Any]]:
    """Process document using unstructured library based on file type."""
    try:
        # Create sanitized base ID from original filename
        base_id = sanitize_id(original_filename)
        log_debug(f"Sanitized ID base: {base_id}")
        
        # Partition the document based on file type
        if file_type == "pdf":
            print("\n=== Starting PDF Processing ===")  # Force print for debugging
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                include_metadata=True
            )
            print(f"\nPDF Partitioning - Number of elements: {len(elements)}")  # Force print
            
            # Log table detection information
            print("\n=== Table Detection Analysis ===")
            print(f"Total elements found: {len(elements)}")
            
            # Get all element types and their counts
            element_types = {}
            table_count = 0
            for i, e in enumerate(elements):
                e_type = type(e).__name__
                element_types[e_type] = element_types.get(e_type, 0) + 1
                
                # Print element info for debugging
                print(f"\nElement {i}:")
                print(f"Type: {e_type}")
                print(f"Has 'type' attribute: {hasattr(e, 'type')}")
                if hasattr(e, 'type'):
                    print(f"Element type value: {e.type}")
                if hasattr(e, 'text'):
                    print(f"Text preview: {e.text[:200]}...")
                
                if hasattr(e, 'type') and e.type == 'Table':
                    table_count += 1
                    print(f"\nDetected Table {table_count}:")
                    print(f"- Content Preview:\n{e.text[:500]}...")
                    if hasattr(e, 'metadata'):
                        print(f"- Table Metadata: {e.metadata}")
            
            print(f"\nElement type summary: {element_types}")
            print(f"Number of explicit tables found: {table_count}")
            
            # Look for table-like structures
            table_markers = ['role', 'location', 'name', 'years', 'experience', 'job title', 'uk', 'poland', 'manager', 'lead', 'architect']
            print("\nAnalyzing for table-like structures...")
            for i, e in enumerate(elements):
                if hasattr(e, 'text'):
                    text = e.text.strip().lower()
                    if any(marker in text for marker in table_markers):
                        print(f"\nPotential table-like structure found in element {i}:")
                        print(f"Text preview: {text[:200]}...")
            
            # Look for explicit tables and table-like structures
            table_elements = []
            current_table = []
            in_table = False
            table_markers = ['role', 'location', 'name', 'years', 'experience', 'job title', 'uk', 'poland', 'manager', 'lead', 'architect']
            
            for i, e in enumerate(elements):
                if not hasattr(e, 'text'):
                    continue
                    
                text = e.text.strip().lower()
                
                # Check for explicit table elements
                if hasattr(e, 'type') and e.type == 'Table':
                    if current_table:
                        combined_text = "\n".join(e.text.strip() for e in current_table)
                        table_elements.append(('structured_table', CompositeElement(
                            text=combined_text,
                            metadata=current_table[0].metadata
                        )))
                        current_table = []
                    table_elements.append(('explicit', e))
                    continue
                
                # Check for table headers
                if not in_table and any(marker in text for marker in ['role within project', 'location', 'name', 'years of experience', 'job title']):
                    in_table = True
                    current_table = [e]
                    continue
                
                # If we're in a table, check if this element belongs to it
                if in_table:
                    # Check if this is likely the end of the table
                    if text.startswith(('annex', 'confidential', 'docusign', '1.', '2.')):
                        if len(current_table) >= 3:  # Minimum rows for a valid table
                            combined_text = "\n".join(e.text.strip() for e in current_table)
                            table_elements.append(('structured_table', CompositeElement(
                                text=combined_text,
                                metadata=current_table[0].metadata
                            )))
                        current_table = []
                        in_table = False
                    # Check if this element contains table-like content
                    elif any(marker in text for marker in table_markers) or len(text.split()) <= 10:
                        current_table.append(e)
                    else:
                        # If we hit a long text that doesn't match table patterns, check if it's just a description
                        next_elem_table_like = False
                        if i + 1 < len(elements) and hasattr(elements[i+1], 'text'):
                            next_text = elements[i+1].text.strip().lower()
                            next_elem_table_like = any(marker in next_text for marker in table_markers)
                        
                        if next_elem_table_like:
                            current_table.append(e)
                        else:
                            if len(current_table) >= 3:
                                combined_text = "\n".join(e.text.strip() for e in current_table)
                                table_elements.append(('structured_table', CompositeElement(
                                    text=combined_text,
                                    metadata=current_table[0].metadata
                                )))
                            current_table = []
                            in_table = False
            
            # Handle any remaining table content
            if current_table and len(current_table) >= 3:
                combined_text = "\n".join(e.text.strip() for e in current_table)
                table_elements.append(('structured_table', CompositeElement(
                    text=combined_text,
                    metadata=current_table[0].metadata
                )))
            
            if table_elements:
                print(f"\nFound {len(table_elements)} potential table structures:")
                for i, (table_type, table) in enumerate(table_elements):
                    print(f"\nTable {i + 1} ({table_type}):")
                    print(f"- Text Preview:\n{table.text[:500]}...")
                    
                    # Analyze table structure
                    lines = table.text.strip().split('\n')
                    if len(lines) > 1:
                        print(f"- Structure Analysis:")
                        print(f"  Total Lines: {len(lines)}")
                        print(f"  Sample Lines:")
                        for j, line in enumerate(lines[:5]):
                            print(f"    {j+1}: {line.strip()}")
            else:
                print("\nNo table structures detected. Analyzing general elements:")
                for i, element in enumerate(elements[:5]):
                    print(f"\nElement {i + 1}:")
                    print(f"- Type: {type(element).__name__}")
                    if hasattr(element, 'text'):
                        print(f"- Text Preview: {element.text[:200]}...")
                    if hasattr(element, 'metadata'):
                        metadata = element.metadata.__dict__ if hasattr(element.metadata, '__dict__') else element.metadata
                        print(f"- Metadata Keys: {metadata.keys()}")
            
            if elements:
                print(f"First element type: {type(elements[0])}")
                print(f"First element metadata: {elements[0].metadata if hasattr(elements[0], 'metadata') else 'No metadata'}")
                print(f"First element attributes: {dir(elements[0])}")
        elif file_type == "docx":
            elements = partition_docx(filename=file_path)
            print(f"\nDOCX Partitioning - Number of elements: {len(elements)}")  # Force print
            if elements:
                print(f"First element type: {type(elements[0])}")
                print(f"First element metadata: {elements[0].metadata if hasattr(elements[0], 'metadata') else 'No metadata'}")
                print(f"First element attributes: {dir(elements[0])}")
        elif file_type == "xlsx":
            elements = partition_xlsx(filename=file_path)
            print(f"\nXLSX Partitioning - Number of elements: {len(elements)}")  # Force print
            if elements:
                print(f"First element type: {type(elements[0])}")
                print(f"First element metadata: {elements[0].metadata if hasattr(elements[0], 'metadata') else 'No metadata'}")
                print(f"First element attributes: {dir(elements[0])}")
        elif file_type == "pptx":
            elements = partition_pptx(filename=file_path)
            print(f"\nPPTX Partitioning - Number of elements: {len(elements)}")  # Force print
            if elements:
                print(f"First element type: {type(elements[0])}")
                print(f"First element metadata: {elements[0].metadata if hasattr(elements[0], 'metadata') else 'No metadata'}")
                print(f"First element attributes: {dir(elements[0])}")
        elif file_type == "txt":
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                elements = [{"text": text}]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Chunk the document using title-based chunking
        chunks = chunk_by_title(
            elements,
            max_characters=8000,
            new_after_n_chars=6000,
        )
        
        # Debug logging
        log_debug(f"Number of chunks created: {len(chunks)}")
        if chunks:
            log_debug(f"First chunk type: {type(chunks[0])}")
            log_debug(f"First chunk content: {chunks[0]}")
            
        # Convert chunks to structured documents
        documents = []
        for i, chunk in enumerate(chunks):
            try:
                # Debug logging for chunk structure
                log_debug(f"Chunk type: {type(chunk)}")
                log_debug(f"Chunk attributes: {dir(chunk) if not isinstance(chunk, (dict, str)) else 'Basic type'}")
                log_debug(f"Raw chunk metadata: {chunk.metadata if hasattr(chunk, 'metadata') else 'No metadata'}")

                # Extract text content from CompositeElement
                if isinstance(chunk, (dict, str)):
                    chunk_text = str(chunk)
                else:
                    # Handle CompositeElement
                    chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)

                # Get metadata
                metadata = {}
                if hasattr(chunk, 'metadata'):
                    metadata = chunk.metadata
                elif isinstance(chunk, dict) and 'metadata' in chunk:
                    metadata = chunk['metadata']

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

                # Create document with required fields
                metadata = {
                    "source": original_filename,
                    "page_numbers": list(sorted(page_numbers)) if page_numbers else None
                }
                
                doc = {
                    "id": f"{base_id}_{i}",
                    "content": chunk_text,
                    "metadata": json.dumps(metadata),  # Serialize metadata as JSON string
                    "source": original_filename  # Add source field here
                }
                log_debug(f"Created document {i} with content length: {len(chunk_text)}")
                
                # Debug log for table chunks
                if hasattr(chunk, 'type') and chunk.type == 'Table':
                    log_debug(f"\nProcessing Table Chunk {i}:")
                    log_debug(f"- Content Length: {len(chunk_text)}")
                    log_debug(f"- Metadata Keys: {chunk.metadata.keys() if hasattr(chunk, 'metadata') else 'No metadata'}")
                
                documents.append(doc)
                
            except Exception as e:
                log_debug(f"Error processing chunk {i}: {str(e)}")
                raise
            
        return documents

    except Exception as e:
        logger.error(f"Error processing {file_type} file: {str(e)}")
        raise

def delete_file_from_index(search_client, source_file: str) -> bool:
    """Delete all documents with the given source file from the index."""
    try:
        # Search for all documents with the given source
        results = search_client.search(
            search_text="*",
            filter=f"source eq '{source_file}'",
            select=["id"],
            include_total_count=True
        )
        
        # Get all document IDs
        doc_ids = [doc["id"] for doc in results]
        
        if not doc_ids:
            logger.warning(f"No documents found for source file: {source_file}")
            return False
        
        # Delete the documents
        result = search_client.delete_documents(documents=[{"@search.action": "delete", "id": doc_id} for doc_id in doc_ids])
        
        # Check if all deletions were successful
        failed_deletions = [r for r in result if r.succeeded is False]
        if failed_deletions:
            logger.error(f"Failed to delete some documents: {failed_deletions}")
            return False
            
        logger.info(f"Successfully deleted {len(doc_ids)} documents for {source_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting file from index: {str(e)}")
        return False

def replace_file_in_index(search_client, vector_store, file_path: str, filename: str, file_extension: str) -> bool:
    """Replace an existing file in the index with a new one."""
    try:
        # First delete the existing file
        logger.info(f"Attempting to delete existing file: {filename}")
        if not delete_file_from_index(search_client, filename):
            logger.error(f"Failed to delete existing file: {filename}")
            return False
        
        # Process the new file
        logger.info(f"Processing new file: {filename}")
        documents = process_document(file_path, filename, file_extension)
        
        # Add new documents to index
        logger.info(f"Adding new documents to index for: {filename}")
        add_documents_to_index(documents, vector_store)
        
        logger.info(f"Successfully replaced {filename} in the index")
        return True
        
    except Exception as e:
        logger.error(f"Error replacing file in index: {str(e)}")
        return False

def main():
    """Main application function."""
    st.title("Document Search App")
    
    try:
        # Handle the clear_upload flag
        if st.session_state.get('clear_upload', False):
            st.session_state.clear_upload = False
            st.session_state.file_uploader = None
            return
        
        # Initialize Azure clients
        search_client = initialize_search_client()
        vector_store = get_vector_store()
        
        # Create sidebar with indexed files
        with st.sidebar:
            st.header("Indexed Files")
            
            # Create a container with fixed height and scrollbar
            with st.container():
                # Apply custom CSS for scrollable container
                st.markdown("""
                    <style>
                        [data-testid="stSidebarContent"] > div:nth-child(2) {
                            max-height: calc(100vh - 100px);
                            overflow-y: auto;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                indexed_files = get_indexed_files(search_client)
                if indexed_files:
                    for file in indexed_files:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(f"{file[0]}")
                        with col2:
                            if st.button("🗑️", key=f"delete_{file[0]}", help=f"Delete {file[0]}"):
                                if st.session_state.get(f'confirm_delete_{file[0]}', False):
                                    # Perform deletion
                                    if delete_file_from_index(search_client, file[0]):
                                        st.success(f"Successfully deleted {file[0]}")
                                        st.rerun()  # Refresh the page to update the sidebar
                                    else:
                                        st.error(f"Failed to delete {file[0]}")
                                    st.session_state[f'confirm_delete_{file[0]}'] = False
                                else:
                                    # Show confirmation
                                    st.session_state[f'confirm_delete_{file[0]}'] = True
                                    st.warning(f"Are you sure you want to delete {file[0]}? Click the delete button again to confirm.")
                else:
                    st.text("No files indexed yet")
        
        # File uploader
        uploaded_files = st.file_uploader("Choose files to upload (PDF, DOCX, XLSX, PPTX, or TXT)", 
                                        accept_multiple_files=True, 
                                        type=['pdf', 'docx', 'txt', 'xlsx', 'pptx'],
                                        key='file_uploader')

        if uploaded_files:
            # Get current indexed files
            indexed_files = get_indexed_files(search_client)
            
            # Check for existing files and show radio buttons if needed
            existing_files = [f for f in uploaded_files if f.name in [file[0] for file in indexed_files]]
            new_files = [f for f in uploaded_files if f.name not in [file[0] for file in indexed_files]]
            
            # Initialize session state for file decisions if not exists
            if 'file_decisions' not in st.session_state:
                st.session_state.file_decisions = {}
            
            # Show decisions for existing files
            if existing_files:
                st.write("The following files already exist in the index:")
                for file in existing_files:
                    if file.name not in st.session_state.file_decisions:
                        st.session_state.file_decisions[file.name] = "replace"  # Default to replace
                    
                    decision = st.radio(
                        f"Action for {file.name}:",
                        options=["replace", "skip"],
                        key=f"radio_{file.name}",
                        index=0 if st.session_state.file_decisions[file.name] == "replace" else 1
                    )
                    st.session_state.file_decisions[file.name] = decision
            
            # Show summary of actions
            if new_files:
                st.write("New files to be uploaded:")
                for file in new_files:
                    st.write(f"- {file.name}")
            
            # Add a process button
            if st.button("Process Files"):
                # Create a progress container
                progress_container = st.empty()
                
                # Calculate total files to process
                files_to_process = new_files + [f for f in existing_files if st.session_state.file_decisions[f.name] == "replace"]
                total_files = len(files_to_process)
                processed_count = 0
                
                # Process new files first
                for uploaded_file in new_files:
                    processed_count += 1
                    try:
                        # Update progress message
                        with progress_container:
                            st.info(f"{uploaded_file.name} being uploaded ({processed_count}/{total_files})")
                        
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            file_path = tmp_file.name

                        # Get the file extension
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        # Process new file
                        with st.spinner(f'Processing {uploaded_file.name} ({processed_count}/{total_files})...'):
                            documents = process_document(file_path, uploaded_file.name, file_extension)
                            add_documents_to_index(documents, vector_store)
                            st.success(f"Successfully processed and indexed {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        if 'file_path' in locals():
                            os.unlink(file_path)
                
                # Process existing files based on decisions
                for uploaded_file in existing_files:
                    if st.session_state.file_decisions[uploaded_file.name] == "replace":
                        processed_count += 1
                        try:
                            # Update progress message
                            with progress_container:
                                st.info(f"{uploaded_file.name} being updated ({processed_count}/{total_files})")
                            
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                file_path = tmp_file.name

                            # Get the file extension
                            file_extension = uploaded_file.name.split('.')[-1].lower()
                            
                            # Replace existing file
                            with st.spinner(f'Updating {uploaded_file.name} ({processed_count}/{total_files})...'):
                                if replace_file_in_index(search_client, vector_store, file_path, uploaded_file.name, file_extension):
                                    st.success(f"Successfully replaced {uploaded_file.name}")
                                else:
                                    st.error(f"Failed to replace {uploaded_file.name}")
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            if 'file_path' in locals():
                                os.unlink(file_path)
                    else:
                        st.info(f"Skipped {uploaded_file.name}")
                
                # Clear the file decisions after processing
                st.session_state.file_decisions = {}
                # Set a flag to trigger page rerun on next load
                st.session_state.clear_upload = True
                st.rerun()
                
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")

if __name__ == "__main__":
    main()